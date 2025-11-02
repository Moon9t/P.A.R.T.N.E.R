package main

import (
	"flag"
	"fmt"
	"log"
	"os"
	"os/signal"
	"path/filepath"
	"syscall"
	"time"

	"github.com/thyrook/partner/internal/vision"
	"gocv.io/x/gocv"
)

func main() {
	// Command line flags
	imageFile := flag.String("image", "", "Path to chess board image file")
	videoFile := flag.String("video", "", "Path to chess game video file")
	liveMode := flag.Bool("live", false, "Enable live screen capture mode")
	configFile := flag.String("config", "vision-config.json", "Path to vision configuration file")
	showWindow := flag.Bool("show", false, "Show visualization window")
	saveOutput := flag.String("output", "", "Save detected board to file")
	verbose := flag.Bool("v", false, "Verbose output")

	flag.Parse()

	// Load or create configuration
	config := loadConfig(*configFile)

	// Execute mode based on flags
	switch {
	case *imageFile != "":
		testSingleImage(*imageFile, config, *showWindow, *saveOutput, *verbose)
	case *videoFile != "":
		testVideoFile(*videoFile, config, *showWindow, *verbose)
	case *liveMode:
		testLiveCapture(config, *showWindow, *verbose)
	default:
		fmt.Println("P.A.R.T.N.E.R Vision Module Test Tool")
		fmt.Println("\nUsage:")
		flag.PrintDefaults()
		os.Exit(1)
	}
}

func loadConfig(path string) *vision.Config {
	if _, err := os.Stat(path); os.IsNotExist(err) {
		log.Printf("Config file not found, using defaults: %s", path)
		return vision.DefaultConfig()
	}

	config, err := vision.LoadConfig(path)
	if err != nil {
		log.Printf("Error loading config, using defaults: %v", err)
		return vision.DefaultConfig()
	}

	if err := config.Validate(); err != nil {
		log.Fatalf("Invalid configuration: %v", err)
	}

	log.Printf("Loaded configuration from %s", path)
	return config
}

func testSingleImage(imagePath string, config *vision.Config, show bool, outputPath string, verbose bool) {
	fmt.Printf("Testing single image: %s\n", imagePath)

	// Load image
	img := gocv.IMRead(imagePath, gocv.IMReadColor)
	if img.Empty() {
		log.Fatalf("Failed to load image: %s", imagePath)
	}
	defer img.Close()

	fmt.Printf("Image loaded: %dx%d\n", img.Cols(), img.Rows())

	// Create detector
	detector := vision.NewBoardDetector(config.SquareSize, config.UseGrayscale)

	// Detect board
	start := time.Now()
	tensor, err := detector.DetectBoard(&img)
	elapsed := time.Since(start)

	if err != nil {
		log.Fatalf("Board detection failed: %v", err)
	}

	fmt.Printf("Detection completed in %v\n", elapsed)

	// Validate tensor
	if err := vision.ValidateBoardTensor(tensor); err != nil {
		log.Printf("WARNING: Board validation failed: %v", err)
	} else {
		fmt.Println("✓ Board tensor validated successfully")
	}

	// Print board
	fmt.Println("\nDetected board position:")
	fmt.Println(vision.PrintBoardTensor(tensor))

	// Count pieces
	whitePieces, blackPieces := countPieces(tensor)
	fmt.Printf("\nPiece count: White=%d, Black=%d, Total=%d\n",
		whitePieces, blackPieces, whitePieces+blackPieces)

	if verbose {
		printTensorStats(tensor)
	}

	// Show visualization if requested
	if show {
		showVisualization(&img, tensor, "Detected Board")
	}

	// Save output if requested
	if outputPath != "" {
		saveDetectedBoard(&img, tensor, outputPath)
	}
}

func testVideoFile(videoPath string, config *vision.Config, show bool, verbose bool) {
	fmt.Printf("Testing video file: %s\n", videoPath)

	// Get video info
	info, err := vision.GetVideoInfo(videoPath)
	if err != nil {
		log.Fatalf("Failed to get video info: %v", err)
	}

	fmt.Printf("Video info:\n")
	fmt.Printf("  Resolution: %dx%d\n", info.Width, info.Height)
	fmt.Printf("  FPS: %.2f\n", info.FPS)
	fmt.Printf("  Duration: %v\n", info.Duration)
	fmt.Printf("  Frames: %d\n", info.FrameCount)

	// Create pipeline with video source
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

	fmt.Println("\nProcessing video (Ctrl+C to stop)...")

	// Setup signal handling
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, os.Interrupt, syscall.SIGTERM)

	// Create window if showing
	var window *gocv.Window
	if show {
		window = gocv.NewWindow("Video Playback")
		defer window.Close()
	}

	// Process tensors
	frameCount := 0
	changeCount := 0
	ticker := time.NewTicker(5 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case tensorData, ok := <-tensorChan:
			if !ok {
				fmt.Println("\nVideo processing complete")
				printFinalStats(pipeline, frameCount, changeCount)
				return
			}

			frameCount++
			if len(tensorData.Changes) > 0 {
				changeCount++
				if verbose {
					fmt.Printf("\n[Frame %d] Board changed:\n", frameCount)
					fmt.Println(vision.PrintBoardTensor(tensorData.Tensor))
				}
			}

		case <-ticker.C:
			stats := pipeline.GetStats()
			printProgressStats(stats, frameCount, changeCount)

		case <-sigChan:
			fmt.Println("\nInterrupted by user")
			printFinalStats(pipeline, frameCount, changeCount)
			return
		}
	}
}

func testLiveCapture(config *vision.Config, show bool, verbose bool) {
	fmt.Println("Testing live screen capture")
	fmt.Printf("Capture region: %d,%d (%dx%d)\n",
		config.CaptureRegion.X, config.CaptureRegion.Y,
		config.CaptureRegion.Width, config.CaptureRegion.Height)
	fmt.Printf("Target FPS: %d\n", config.FPS)

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

	fmt.Println("\nCapturing screen (Ctrl+C to stop)...")
	fmt.Println("Move chess pieces to trigger detection")

	// Setup signal handling
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, os.Interrupt, syscall.SIGTERM)

	// Create window if showing
	var window *gocv.Window
	if show {
		window = gocv.NewWindow("Live Capture")
		defer window.Close()
	}

	// Process tensors
	frameCount := 0
	changeCount := 0
	ticker := time.NewTicker(5 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case tensorData := <-tensorChan:
			frameCount++
			if len(tensorData.Changes) > 0 {
				changeCount++
				fmt.Printf("\n[Change %d detected at frame %d]\n", changeCount, frameCount)
				fmt.Println(vision.PrintBoardTensor(tensorData.Tensor))

				// Validate
				if err := vision.ValidateBoardTensor(tensorData.Tensor); err != nil {
					log.Printf("Warning: Invalid board state: %v", err)
				}
			} else if verbose {
				fmt.Printf("Frame %d: No change\n", frameCount)
			}

		case <-ticker.C:
			stats := pipeline.GetStats()
			printProgressStats(stats, frameCount, changeCount)

		case <-sigChan:
			fmt.Println("\nStopping capture...")
			printFinalStats(pipeline, frameCount, changeCount)
			return
		}
	}
}

func countPieces(tensor [12][8][8]float32) (white, black int) {
	for channel := 0; channel < 12; channel++ {
		for rank := 0; rank < 8; rank++ {
			for file := 0; file < 8; file++ {
				if tensor[channel][rank][file] > 0.5 {
					if channel < 6 {
						white++
					} else {
						black++
					}
				}
			}
		}
	}
	return
}

func printTensorStats(tensor [12][8][8]float32) {
	fmt.Println("\nPer-channel piece count:")
	pieceNames := []string{
		"White Pawn", "White Knight", "White Bishop", "White Rook", "White Queen", "White King",
		"Black Pawn", "Black Knight", "Black Bishop", "Black Rook", "Black Queen", "Black King",
	}

	for channel := 0; channel < 12; channel++ {
		count := 0
		for rank := 0; rank < 8; rank++ {
			for file := 0; file < 8; file++ {
				if tensor[channel][rank][file] > 0.5 {
					count++
				}
			}
		}
		if count > 0 {
			fmt.Printf("  %s: %d\n", pieceNames[channel], count)
		}
	}
}

func printProgressStats(stats vision.PipelineStats, frames, changes int) {
	fmt.Printf("\n[Stats] Frames: %d | Changes: %d | Errors: %d | Last: %v\n",
		frames, changes, stats.Errors, stats.LastProcessTime)
}

func printFinalStats(pipeline *vision.Pipeline, frames, changes int) {
	stats := pipeline.GetStats()
	fmt.Printf("\n=== Final Statistics ===\n")
	fmt.Printf("Total frames processed: %d\n", frames)
	fmt.Printf("Board changes detected: %d\n", changes)
	fmt.Printf("Errors encountered: %d\n", stats.Errors)
	if frames > 0 {
		fmt.Printf("Change rate: %.2f%%\n", float64(changes)/float64(frames)*100)
	}
	fmt.Printf("Average processing time: %v\n", stats.LastProcessTime)
}

func showVisualization(img *gocv.Mat, tensor [12][8][8]float32, title string) {
	window := gocv.NewWindow(title)
	defer window.Close()

	// Show original image
	window.IMShow(*img)
	window.WaitKey(0)
}

func saveDetectedBoard(img *gocv.Mat, tensor [12][8][8]float32, outputPath string) {
	// Ensure output directory exists
	dir := filepath.Dir(outputPath)
	if err := os.MkdirAll(dir, 0755); err != nil {
		log.Printf("Failed to create output directory: %v", err)
		return
	}

	// Save original image
	if ok := gocv.IMWrite(outputPath, *img); !ok {
		log.Printf("Failed to save image to %s", outputPath)
		return
	}

	fmt.Printf("Saved detected board to: %s\n", outputPath)

	// Save tensor as text
	textPath := outputPath + ".txt"
	f, err := os.Create(textPath)
	if err != nil {
		log.Printf("Failed to create text file: %v", err)
		return
	}
	defer f.Close()

	f.WriteString(vision.PrintBoardTensor(tensor))
	fmt.Printf("Saved board position to: %s\n", textPath)
}
