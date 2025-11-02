package main

import (
	"flag"
	"fmt"
	"math/rand"
	"os"
	"time"

	"github.com/thyrook/partner/internal/config"
	"github.com/thyrook/partner/internal/model"
	"github.com/thyrook/partner/internal/vision"
)

func main() {
	// Parse flags
	configPath := flag.String("config", "config.json", "Path to config file")
	testMode := flag.String("mode", "synthetic", "Test mode: synthetic, vision, or benchmark")
	iterations := flag.Int("iterations", 100, "Number of iterations for benchmark")
	flag.Parse()

	fmt.Println("═══════════════════════════════════════════════════════════")
	fmt.Println("  P.A.R.T.N.E.R - CNN Model Testing (Phase 3)")
	fmt.Println("═══════════════════════════════════════════════════════════")
	fmt.Println()

	// Load configuration
	cfg, err := config.Load(*configPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "❌ Failed to load config: %v\n", err)
		os.Exit(1)
	}

	// Initialize model
	fmt.Println("Initializing CNN Model...")
	fmt.Println("───────────────────────────────────────────────────────────")
	fmt.Printf("  Architecture:\n")
	fmt.Printf("    Input Layer:  %dx%d grayscale (64 values)\n", 8, 8)
	fmt.Printf("    Conv1:        1→16 channels, 3x3 kernel, ReLU\n")
	fmt.Printf("    MaxPool1:     2x2\n")
	fmt.Printf("    Conv2:        16→32 channels, 3x3 kernel, ReLU\n")
	fmt.Printf("    MaxPool2:     2x2\n")
	fmt.Printf("    Flatten:      32*2*2 = 128 features\n")
	fmt.Printf("    FC1:          128→%d, ReLU\n", cfg.Model.HiddenSize)
	fmt.Printf("    FC2:          %d→%d, ReLU\n", cfg.Model.HiddenSize, cfg.Model.HiddenSize)
	fmt.Printf("    Output:       %d→%d move probabilities (Softmax)\n", cfg.Model.HiddenSize, cfg.Model.OutputSize)
	fmt.Println()

	net, err := model.NewChessNet(cfg.Model.InputSize, cfg.Model.HiddenSize, cfg.Model.OutputSize)
	if err != nil {
		fmt.Fprintf(os.Stderr, "❌ Failed to create model: %v\n", err)
		os.Exit(1)
	}
	defer net.Close()

	// Count parameters
	params := countParameters(cfg)
	fmt.Printf("  Total Parameters: %d (%.2f KB)\n", params, float64(params*8)/1024)
	fmt.Println("✓ Model initialized")
	fmt.Println()

	// Load existing model if available
	if model.ModelExists(cfg.Model.ModelPath) {
		fmt.Printf("Loading model weights from %s...\n", cfg.Model.ModelPath)
		if err := net.Load(cfg.Model.ModelPath); err != nil {
			fmt.Printf("⚠️  Failed to load model: %v\n", err)
			fmt.Println("Continuing with random weights...")
		} else {
			fmt.Println("✓ Model loaded successfully")
		}
		fmt.Println()
	}

	// Run selected test mode
	switch *testMode {
	case "synthetic":
		runSyntheticTest(net, cfg)
	case "vision":
		runVisionTest(net, cfg)
	case "benchmark":
		runBenchmark(net, cfg, *iterations)
	default:
		fmt.Fprintf(os.Stderr, "❌ Unknown test mode: %s\n", *testMode)
		os.Exit(1)
	}
}

// runSyntheticTest tests the model with synthetic board states
func runSyntheticTest(net *model.ChessNet, cfg *config.Config) {
	fmt.Println("Running Synthetic Data Test")
	fmt.Println("═══════════════════════════════════════════════════════════")
	fmt.Println()

	testCases := []struct {
		name        string
		description string
		generator   func() []float64
	}{
		{
			name:        "Empty Board",
			description: "All zeros (empty squares)",
			generator:   func() []float64 { return make([]float64, 64) },
		},
		{
			name:        "Full Board",
			description: "All ones (occupied squares)",
			generator: func() []float64 {
				board := make([]float64, 64)
				for i := range board {
					board[i] = 1.0
				}
				return board
			},
		},
		{
			name:        "Starting Position",
			description: "Simulated chess starting position",
			generator:   generateStartingPosition,
		},
		{
			name:        "Random Position",
			description: "Random piece placement",
			generator:   generateRandomPosition,
		},
	}

	for i, tc := range testCases {
		fmt.Printf("%d. %s\n", i+1, tc.name)
		fmt.Printf("   Description: %s\n", tc.description)

		// Generate board
		board := tc.generator()

		// Visualize
		fmt.Println("   Board State:")
		fmt.Print(vision.VisualizeBoard(board, 8))

		// Run inference
		start := time.Now()
		predictions, err := net.Predict(board)
		duration := time.Since(start)

		if err != nil {
			fmt.Printf("   ❌ Prediction failed: %v\n\n", err)
			continue
		}

		// Analyze predictions
		topMoves := model.GetTopKMoves(predictions, 5)

		fmt.Printf("   Inference Time: %v\n", duration)
		fmt.Printf("   Top 5 Predicted Moves:\n")
		for j, move := range topMoves {
			moveNotation := model.DecodeMove(move.MoveIndex)
			fmt.Printf("      %d. %s (%.4f confidence)\n", j+1, moveNotation, move.Score)
		}

		// Statistics
		sum := 0.0
		max := 0.0
		for _, p := range predictions {
			sum += p
			if p > max {
				max = p
			}
		}
		fmt.Printf("   Probability Sum: %.6f (should be ~1.0)\n", sum)
		fmt.Printf("   Max Probability: %.6f\n", max)
		fmt.Println()
	}

	fmt.Println("═══════════════════════════════════════════════════════════")
	fmt.Println("✓ Synthetic test complete")
}

// runVisionTest tests the model with actual vision capture
func runVisionTest(net *model.ChessNet, cfg *config.Config) {
	fmt.Println("Running Vision Integration Test")
	fmt.Println("═══════════════════════════════════════════════════════════")
	fmt.Println()

	// Initialize vision capturer
	capturer := vision.NewCapturer(
		cfg.Vision.ScreenRegion.X,
		cfg.Vision.ScreenRegion.Y,
		cfg.Vision.ScreenRegion.Width,
		cfg.Vision.ScreenRegion.Height,
		cfg.Vision.BoardSize,
		cfg.Vision.DiffThreshold,
	)
	defer capturer.Close()

	fmt.Println("Testing vision capture...")
	if err := capturer.ValidateCapture(); err != nil {
		fmt.Fprintf(os.Stderr, "❌ Vision validation failed: %v\n", err)
		os.Exit(1)
	}
	fmt.Println("✓ Vision capture validated")
	fmt.Println()

	fmt.Println("Capturing and analyzing 5 frames...")
	fmt.Println()

	for i := 0; i < 5; i++ {
		fmt.Printf("Frame %d:\n", i+1)

		// Capture and process
		state, err := capturer.ExtractBoardState()
		if err != nil {
			fmt.Printf("  ❌ Capture failed: %v\n\n", err)
			continue
		}

		fmt.Printf("  Timestamp: %s\n", state.Timestamp.Format("15:04:05.000"))
		fmt.Printf("  Changed: %v (diff: %.2f)\n", state.Changed, state.DiffScore)

		// Visualize board
		fmt.Println("  Board State:")
		fmt.Print("  " + vision.VisualizeBoard(state.Grid, 8))

		// Run inference
		start := time.Now()
		predictions, err := net.Predict(state.Grid)
		duration := time.Since(start)

		if err != nil {
			fmt.Printf("  ❌ Prediction failed: %v\n\n", err)
			continue
		}

		// Show top moves
		topMoves := model.GetTopKMoves(predictions, 3)
		fmt.Printf("  Inference Time: %v\n", duration)
		fmt.Printf("  Top 3 Moves:\n")
		for j, move := range topMoves {
			moveNotation := model.DecodeMove(move.MoveIndex)
			fmt.Printf("    %d. %s (%.4f)\n", j+1, moveNotation, move.Score)
		}
		fmt.Println()

		// Wait before next capture
		if i < 4 {
			time.Sleep(500 * time.Millisecond)
		}
	}

	fmt.Println("═══════════════════════════════════════════════════════════")
	fmt.Println("✓ Vision integration test complete")
}

// runBenchmark benchmarks the model inference performance
func runBenchmark(net *model.ChessNet, cfg *config.Config, iterations int) {
	fmt.Println("Running Performance Benchmark")
	fmt.Println("═══════════════════════════════════════════════════════════")
	fmt.Println()

	fmt.Printf("Iterations: %d\n", iterations)
	fmt.Println()

	// Generate random boards
	boards := make([][]float64, iterations)
	for i := 0; i < iterations; i++ {
		boards[i] = generateRandomPosition()
	}

	// Warmup
	fmt.Println("Warming up...")
	for i := 0; i < 10; i++ {
		_, _ = net.Predict(boards[i%len(boards)])
	}
	fmt.Println("✓ Warmup complete")
	fmt.Println()

	// Benchmark
	fmt.Println("Running benchmark...")
	start := time.Now()

	for i := 0; i < iterations; i++ {
		_, err := net.Predict(boards[i])
		if err != nil {
			fmt.Fprintf(os.Stderr, "❌ Prediction failed at iteration %d: %v\n", i, err)
			os.Exit(1)
		}

		if (i+1)%10 == 0 {
			elapsed := time.Since(start)
			rate := float64(i+1) / elapsed.Seconds()
			fmt.Printf("\rProgress: %d/%d (%.1f inferences/sec)", i+1, iterations, rate)
		}
	}

	duration := time.Since(start)
	fmt.Println()
	fmt.Println()

	// Results
	avgTime := duration / time.Duration(iterations)
	fps := float64(iterations) / duration.Seconds()

	fmt.Println("Benchmark Results:")
	fmt.Println("───────────────────────────────────────────────────────────")
	fmt.Printf("  Total Time:       %v\n", duration)
	fmt.Printf("  Avg Time/Inference: %v\n", avgTime)
	fmt.Printf("  Throughput:       %.2f inferences/sec\n", fps)
	fmt.Printf("  Latency:          %.2f ms/inference\n", avgTime.Seconds()*1000)
	fmt.Println()

	// Memory statistics
	params := countParameters(cfg)
	modelSize := float64(params*8) / (1024 * 1024) // MB
	fmt.Printf("  Model Size:       %.2f MB\n", modelSize)
	fmt.Printf("  Parameters:       %d\n", params)
	fmt.Println()

	fmt.Println("═══════════════════════════════════════════════════════════")
	fmt.Println("✓ Benchmark complete")
}

// generateStartingPosition creates a simplified starting chess position
func generateStartingPosition() []float64 {
	board := make([]float64, 64)

	// Black pieces (darker) on top two rows
	for i := 0; i < 16; i++ {
		board[i] = 0.8 + rand.Float64()*0.2
	}

	// Empty squares in middle
	for i := 16; i < 48; i++ {
		board[i] = 0.0
	}

	// White pieces (lighter) on bottom two rows
	for i := 48; i < 64; i++ {
		board[i] = 0.2 + rand.Float64()*0.3
	}

	return board
}

// generateRandomPosition creates a random board position
func generateRandomPosition() []float64 {
	board := make([]float64, 64)
	numPieces := 16 + rand.Intn(16) // 16-32 pieces

	for i := 0; i < numPieces; i++ {
		square := rand.Intn(64)
		if rand.Float64() < 0.5 {
			board[square] = 0.2 + rand.Float64()*0.3 // White
		} else {
			board[square] = 0.7 + rand.Float64()*0.3 // Black
		}
	}

	return board
}

// countParameters estimates total model parameters
func countParameters(cfg *config.Config) int {
	// Conv1: (3*3*1 + 1) * 16 = 160
	conv1 := (3*3*1 + 1) * 16

	// Conv2: (3*3*16 + 1) * 32 = 4640
	conv2 := (3*3*16 + 1) * 32

	// FC1: (128 + 1) * hiddenSize
	fc1 := (128 + 1) * cfg.Model.HiddenSize

	// FC2: (hiddenSize + 1) * hiddenSize
	fc2 := (cfg.Model.HiddenSize + 1) * cfg.Model.HiddenSize

	// FC3: (hiddenSize + 1) * outputSize
	fc3 := (cfg.Model.HiddenSize + 1) * cfg.Model.OutputSize

	return conv1 + conv2 + fc1 + fc2 + fc3
}

func init() {
	rand.Seed(time.Now().UnixNano())
}
