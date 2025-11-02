package main

import (
	"flag"
	"fmt"
	"log"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/thyrook/partner/internal/model"
	"github.com/thyrook/partner/internal/vision"
)

func main() {
	// Command line flags
	modelPath := flag.String("model", "data/chess_model.bin", "Path to trained model checkpoint")
	configPath := flag.String("config", "vision-config.json", "Path to vision configuration")
	liveMode := flag.Bool("live", false, "Enable live screen capture mode")
	videoPath := flag.String("video", "", "Path to video file for replay mode")
	imagePath := flag.String("image", "", "Path to single image for analysis")
	topK := flag.Int("top", 5, "Number of top moves to display")
	verbose := flag.Bool("v", false, "Verbose output")

	flag.Parse()

	fmt.Println("=== P.A.R.T.N.E.R Live Chess Analysis ===")
	fmt.Println("Predictive Adaptive Real-Time Neural Evaluation & Response")
	fmt.Println()

	// Load vision configuration
	var config *vision.Config
	var err error

	if _, statErr := os.Stat(*configPath); os.IsNotExist(statErr) {
		log.Printf("Config file not found, using defaults: %s", *configPath)
		config = vision.DefaultConfig()
	} else {
		config, err = vision.LoadConfig(*configPath)
		if err != nil {
			log.Printf("Error loading config, using defaults: %v", err)
			config = vision.DefaultConfig()
		}
	}

	if *verbose {
		fmt.Println("Vision Configuration:")
		fmt.Println(config.String())
	}

	// Load trained model
	fmt.Printf("Loading model from: %s\n", *modelPath)
	cnn, err := model.NewChessCNNForInference(*modelPath)
	if err != nil {
		log.Fatalf("Failed to load model: %v", err)
	}
	fmt.Println("✅ Model loaded successfully")

	// Determine mode
	switch {
	case *imagePath != "":
		analyzeSingleImage(*imagePath, config, cnn, *topK, *verbose)
	case *videoPath != "":
		analyzeVideo(*videoPath, config, cnn, *topK, *verbose)
	case *liveMode:
		analyzeLive(config, cnn, *topK, *verbose)
	default:
		fmt.Println("\nUsage: Specify one of the following modes:")
		fmt.Println("  -image <path>  : Analyze a single chess board image")
		fmt.Println("  -video <path>  : Analyze a recorded game video")
		fmt.Println("  -live          : Analyze live screen capture")
		fmt.Println("\nOptions:")
		flag.PrintDefaults()
		os.Exit(1)
	}
}

func analyzeSingleImage(imagePath string, config *vision.Config, cnn *model.ChessCNN, topK int, verbose bool) {
	fmt.Printf("\n📷 Analyzing image: %s\n", imagePath)

	// Create a temporary pipeline to process the image
	tensorChan := make(chan vision.BoardStateTensor, 1)
	pipeline, err := vision.NewPipeline(config, tensorChan)
	if err != nil {
		log.Fatalf("Failed to create pipeline: %v", err)
	}

	// Process single image
	tensorData, err := pipeline.ProcessSingleImage(imagePath)
	if err != nil {
		log.Fatalf("Failed to process image: %v", err)
	}

	tensor := tensorData.Tensor

	// Validate board
	if err := vision.ValidateBoardTensor(tensor); err != nil {
		log.Printf("⚠️  Warning: Board validation failed: %v", err)
	} else {
		fmt.Println("✅ Board state validated")
	}

	// Display board
	fmt.Println("\nDetected Board Position:")
	fmt.Println(vision.PrintBoardTensor(tensor))

	// Get predictions
	fmt.Println("\n🤔 Analyzing position...")
	predictions, err := predictMoves(cnn, tensor, topK)
	if err != nil {
		log.Fatalf("Prediction failed: %v", err)
	}

	// Display results
	fmt.Printf("\n🎯 Top %d Move Suggestions:\n", topK)
	for i, pred := range predictions {
		fmt.Printf("%d. %s (confidence: %.2f%%)\n", i+1, pred.Move, pred.Confidence*100)
	}
}

func analyzeVideo(videoPath string, config *vision.Config, cnn *model.ChessCNN, topK int, verbose bool) {
	fmt.Printf("\n🎬 Analyzing video: %s\n", videoPath)

	// Get video info
	info, err := vision.GetVideoInfo(videoPath)
	if err != nil {
		log.Fatalf("Failed to get video info: %v", err)
	}

	fmt.Printf("Video: %dx%d @ %.1f FPS, %d frames\n",
		info.Width, info.Height, info.FPS, info.FrameCount)

	// Create pipeline
	tensorChan := make(chan vision.BoardStateTensor, 10)
	pipeline, err := vision.NewPipelineWithVideo(config, videoPath, tensorChan)
	if err != nil {
		log.Fatalf("Failed to create pipeline: %v", err)
	}

	// Start pipeline
	if err := pipeline.Start(); err != nil {
		log.Fatalf("Failed to start pipeline: %v", err)
	}
	defer pipeline.Stop()

	fmt.Println()
	fmt.Println("🎯 Processing video (Ctrl+C to stop)...")
	fmt.Println()

	// Setup signal handling
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, os.Interrupt, syscall.SIGTERM)

	positionCount := 0
	lastPredictionTime := time.Now()

	for {
		select {
		case tensorData, ok := <-tensorChan:
			if !ok {
				fmt.Println("\n✅ Video processing complete")
				stats := pipeline.GetStats()
				fmt.Printf("\nStatistics:\n")
				fmt.Printf("  Frames processed: %d\n", stats.FramesProcessed)
				fmt.Printf("  Positions analyzed: %d\n", positionCount)
				return
			}

			// Only analyze when board changes
			if len(tensorData.Changes) > 0 {
				positionCount++

				// Throttle predictions (max 1 per second)
				if time.Since(lastPredictionTime) < time.Second {
					continue
				}
				lastPredictionTime = time.Now()

				fmt.Printf("\n[Position %d] Board changed:\n", positionCount)
				if verbose {
					fmt.Println(vision.PrintBoardTensor(tensorData.Tensor))
				}

				// Get predictions
				predictions, err := predictMoves(cnn, tensorData.Tensor, topK)
				if err != nil {
					log.Printf("Prediction failed: %v", err)
					continue
				}

				fmt.Printf("Suggested moves: ")
				for i, pred := range predictions {
					if i > 0 {
						fmt.Printf(", ")
					}
					fmt.Printf("%s (%.1f%%)", pred.Move, pred.Confidence*100)
				}
				fmt.Println()
			}

		case <-sigChan:
			fmt.Println("\n\n⏸️  Interrupted by user")
			stats := pipeline.GetStats()
			fmt.Printf("\nStatistics:\n")
			fmt.Printf("  Frames processed: %d\n", stats.FramesProcessed)
			fmt.Printf("  Positions analyzed: %d\n", positionCount)
			return
		}
	}
}

func analyzeLive(config *vision.Config, cnn *model.ChessCNN, topK int, verbose bool) {
	fmt.Println("\n📡 Starting live chess analysis")
	fmt.Printf("Capture region: %d,%d (%dx%d)\n",
		config.CaptureRegion.X, config.CaptureRegion.Y,
		config.CaptureRegion.Width, config.CaptureRegion.Height)
	fmt.Printf("FPS: %d\n", config.FPS)

	// Create pipeline
	tensorChan := make(chan vision.BoardStateTensor, 10)
	pipeline, err := vision.NewPipeline(config, tensorChan)
	if err != nil {
		log.Fatalf("Failed to create pipeline: %v", err)
	}

	// Start pipeline
	if err := pipeline.Start(); err != nil {
		log.Fatalf("Failed to start pipeline: %v", err)
	}
	defer pipeline.Stop()

	fmt.Println()
	fmt.Println("🎯 Watching for moves (Ctrl+C to stop)...")
	fmt.Println()
	fmt.Println("Make a move on the chessboard to see suggestions!")
	fmt.Println()

	// Setup signal handling
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, os.Interrupt, syscall.SIGTERM)

	positionCount := 0
	lastPredictionTime := time.Now()

	for {
		select {
		case tensorData := <-tensorChan:
			// Only analyze when board changes
			if len(tensorData.Changes) > 0 {
				positionCount++

				// Throttle predictions (max 1 per 2 seconds for live mode)
				if time.Since(lastPredictionTime) < 2*time.Second {
					continue
				}
				lastPredictionTime = time.Now()

				fmt.Printf("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
				fmt.Printf("🔄 Move detected! (Position %d)\n", positionCount)
				fmt.Printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n")

				if verbose {
					fmt.Println("Current position:")
					fmt.Println(vision.PrintBoardTensor(tensorData.Tensor))
				}

				// Validate
				if err := vision.ValidateBoardTensor(tensorData.Tensor); err != nil {
					log.Printf("⚠️  Warning: Invalid board state: %v", err)
					continue
				}

				// Get predictions
				fmt.Println("🤔 Analyzing position...")
				predictions, err := predictMoves(cnn, tensorData.Tensor, topK)
				if err != nil {
					log.Printf("❌ Prediction failed: %v", err)
					continue
				}

				// Display results
				fmt.Printf("\n✨ Top %d Move Suggestions:\n", len(predictions))
				for i, pred := range predictions {
					fmt.Printf("  %d. %-6s (%.1f%% confidence)\n",
						i+1, pred.Move, pred.Confidence*100)
				}
				fmt.Println()
			}

		case <-sigChan:
			fmt.Println("\n\n⏸️  Stopping analysis...")
			stats := pipeline.GetStats()
			fmt.Printf("\n📊 Session Statistics:\n")
			fmt.Printf("  Frames processed: %d\n", stats.FramesProcessed)
			fmt.Printf("  Positions analyzed: %d\n", positionCount)
			fmt.Printf("  Average frame time: %v\n", stats.AverageFrameTime)
			if stats.Errors > 0 {
				fmt.Printf("  Errors: %d\n", stats.Errors)
			}
			fmt.Println("\n✅ Analysis complete. Thanks for using P.A.R.T.N.E.R!")
			return
		}
	}
}

// MovePrediction represents a predicted move with confidence
type MovePrediction struct {
	Move       string
	Confidence float64
	FromSquare int
	ToSquare   int
}

func predictMoves(cnn *model.ChessCNN, tensor [12][8][8]float32, topK int) ([]MovePrediction, error) {
	// Extract chess-specific features
	features := model.ExtractChessFeatures(tensor)

	// Get more predictions than needed for filtering
	predictions, err := cnn.Predict(tensor, topK*3)
	if err != nil {
		return nil, fmt.Errorf("model prediction failed: %w", err)
	}

	// Convert to model.MovePrediction format for filtering
	modelPreds := make([]model.MovePrediction, len(predictions))
	for i, pred := range predictions {
		modelPreds[i] = pred
	}

	// Filter out illegal moves
	legalPreds := model.FilterIllegalMoves(modelPreds, tensor)

	// Take top K after filtering
	if len(legalPreds) > topK {
		legalPreds = legalPreds[:topK]
	}

	// Convert to our local format with enhanced notation
	result := make([]MovePrediction, len(legalPreds))
	for i, pred := range legalPreds {
		// Determine piece type
		fromRow, fromCol := pred.FromSquare/8, pred.FromSquare%8
		toRow, toCol := pred.ToSquare/8, pred.ToSquare%8

		pieceType := ""
		isCapture := false

		// Find piece at source
		for ch := 0; ch < 12; ch++ {
			if tensor[ch][fromRow][fromCol] > 0 {
				switch ch % 6 {
				case 0:
					pieceType = "" // Pawn
				case 1:
					pieceType = "N" // Knight
				case 2:
					pieceType = "B" // Bishop
				case 3:
					pieceType = "R" // Rook
				case 4:
					pieceType = "Q" // Queen
				case 5:
					pieceType = "K" // King
				}
				break
			}
		}

		// Check if capture
		for ch := 0; ch < 12; ch++ {
			if tensor[ch][toRow][toCol] > 0 {
				isCapture = true
				break
			}
		}

		// Build move notation
		moveStr := pieceType
		if isCapture && pieceType == "" {
			// Pawn capture - include file
			moveStr = string(rune('a' + fromCol))
		}
		if isCapture {
			moveStr += "x"
		}
		moveStr += indexToSquare(pred.ToSquare)

		// Add context based on features
		context := ""

		if features.MaterialBalance > 0.3 {
			context = " [+material]"
		} else if features.MaterialBalance < -0.3 {
			context = " [-material]"
		}

		if features.GamePhase > 0.7 {
			context += " [endgame]"
		} else if features.GamePhase < 0.3 {
			context += " [opening]"
		}

		// Check if move affects center
		if (toRow == 3 || toRow == 4) && (toCol == 3 || toCol == 4) {
			context += " [center]"
		}

		result[i] = MovePrediction{
			Move:       moveStr + context,
			Confidence: pred.Probability,
			FromSquare: pred.FromSquare,
			ToSquare:   pred.ToSquare,
		}
	}

	return result, nil
}

func indexToSquare(index int) string {
	rank := index / 8
	file := index % 8
	return string(rune('a'+file)) + string(rune('1'+rank))
}
