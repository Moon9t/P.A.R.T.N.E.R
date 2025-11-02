package main

import (
	"flag"
	"fmt"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/thyrook/partner/internal/config"
	"github.com/thyrook/partner/internal/storage"
	"github.com/thyrook/partner/internal/vision"
)

func main() {
	// Parse command line flags
	configPath := flag.String("config", "config.json", "Path to configuration file")
	numSamples := flag.Int("samples", 0, "Target number of samples (0 = unlimited)")
	captureFPS := flag.Int("fps", 2, "Capture rate (frames per second)")
	clearData := flag.Bool("clear", false, "Clear existing data before starting")
	flag.Parse()

	fmt.Println("═══════════════════════════════════════════════════════════")
	fmt.Println("  P.A.R.T.N.E.R - Real Chess Observation Collection")
	fmt.Println("═══════════════════════════════════════════════════════════")
	fmt.Println()

	// Load configuration
	cfg, err := config.Load(*configPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "❌ Failed to load config: %v\n", err)
		os.Exit(1)
	}

	// Ensure data directory exists
	if err := os.MkdirAll("data", 0755); err != nil {
		fmt.Fprintf(os.Stderr, "❌ Failed to create data directory: %v\n", err)
		os.Exit(1)
	}

	// Initialize observation store
	store, err := storage.NewObservationStore(cfg.Training.DBPath, cfg.Training.ReplayBufferSize)
	if err != nil {
		fmt.Fprintf(os.Stderr, "❌ Failed to open observation store: %v\n", err)
		os.Exit(1)
	}
	defer store.Close()

	fmt.Printf("✓ Storage: %s (capacity: %d)\n", cfg.Training.DBPath, cfg.Training.ReplayBufferSize)

	// Display current stats
	displayStats(store)
	fmt.Println()

	// Clear if requested
	if *clearData {
		fmt.Println("⚠️  Clearing existing data...")
		if err := store.Clear(); err != nil {
			fmt.Fprintf(os.Stderr, "❌ Failed to clear store: %v\n", err)
			os.Exit(1)
		}
		fmt.Println("✓ Data cleared")
		fmt.Println()
	}

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

	// Validate capture system
	fmt.Println("Validating screen capture system...")
	if err := capturer.ValidateCapture(); err != nil {
		fmt.Fprintf(os.Stderr, "❌ Screen capture validation failed: %v\n", err)
		fmt.Println()
		fmt.Println("Please ensure:")
		fmt.Println("  1. Your chess board is visible on screen")
		fmt.Println("  2. config.json has correct screen region coordinates")
		fmt.Println("  3. You have screen capture permissions")
		os.Exit(1)
	}
	fmt.Println("✓ Screen capture system ready")
	fmt.Println()

	// Run collection
	runCollection(capturer, store, *numSamples, *captureFPS)
}

func runCollection(capturer *vision.Capturer, store *storage.ObservationStore, targetSamples int, fps int) {
	fmt.Println("═══════════════════════════════════════════════════════════")
	fmt.Println("  Starting Real-Time Observation Collection")
	fmt.Println("═══════════════════════════════════════════════════════════")
	fmt.Println()
	fmt.Println("Instructions:")
	fmt.Println("  1. Open your chess board in the configured screen region")
	fmt.Println("  2. Play chess (online or vs computer)")
	fmt.Println("  3. The system will detect and record each move")
	fmt.Println("  4. Press Ctrl+C when done collecting")
	fmt.Println()

	if targetSamples > 0 {
		fmt.Printf("Target: %d samples\n", targetSamples)
	} else {
		fmt.Println("Target: Unlimited (run until Ctrl+C)")
	}
	fmt.Printf("Capture Rate: %d FPS\n", fps)
	fmt.Println()
	fmt.Println("Monitoring board changes...")
	fmt.Println()

	// Setup signal handling
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, os.Interrupt, syscall.SIGTERM)

	// Collection state
	collected := 0
	startTime := time.Now()
	lastState := []float64{}
	movesDetected := 0
	checksPerSec := 0

	// Ticker for capture rate
	tickerInterval := time.Second / time.Duration(fps)
	ticker := time.NewTicker(tickerInterval)
	defer ticker.Stop()

	// Stats ticker
	statsTicker := time.NewTicker(5 * time.Second)
	defer statsTicker.Stop()

	fmt.Println("Status:")
	fmt.Println("───────────────────────────────────────────────────────────")

	for {
		select {
		case <-sigChan:
			fmt.Println("\n⚠️  Interrupted by user")
			goto done

		case <-statsTicker.C:
			// Periodic stats update
			elapsed := time.Since(startTime)
			rate := float64(collected) / elapsed.Seconds()
			fmt.Printf("\r[%s] Collected: %d | Checks: %d | Rate: %.2f samples/sec          ",
				time.Now().Format("15:04:05"),
				collected,
				checksPerSec,
				rate)

		case <-ticker.C:
			checksPerSec++

			// Capture current board state
			boardState, err := capturer.ExtractBoardState()
			if err != nil {
				// Silently continue - most errors are transient
				continue
			}

			// Check if board actually changed
			if !boardState.Changed {
				continue
			}

			// Detect move by comparing states
			if len(lastState) == 0 {
				// First capture - store initial state
				lastState = make([]float64, len(boardState.Grid))
				copy(lastState, boardState.Grid)
				fmt.Printf("\n✓ Initial board state captured (64 squares)\n")
				continue
			}

			// Board changed - detect the move
			move, err := detectMove(lastState, boardState.Grid)
			if err != nil {
				fmt.Printf("\n⚠️  Could not detect move: %v\n", err)
				continue
			}

			movesDetected++

			// Store the observation (previous state + move that was made)
			if err := store.StoreSample(lastState, move); err != nil {
				fmt.Fprintf(os.Stderr, "\n❌ Failed to store sample: %v\n", err)
				continue
			}

			collected++

			// Visual feedback
			fromSquare := move / 64
			toSquare := move % 64
			fromNotation := squareToNotation(fromSquare)
			toNotation := squareToNotation(toSquare)

			fmt.Printf("\n[%s] Move #%d detected: %s→%s (move index: %d)\n",
				time.Now().Format("15:04:05"),
				movesDetected,
				fromNotation,
				toNotation,
				move)

			// Update last state
			copy(lastState, boardState.Grid)

			// Check if we've reached target
			if targetSamples > 0 && collected >= targetSamples {
				fmt.Printf("\n✓ Target reached: %d samples collected\n", collected)
				goto done
			}
		}
	}

done:
	duration := time.Since(startTime)

	fmt.Println()
	fmt.Println("═══════════════════════════════════════════════════════════")
	fmt.Println("  Collection Complete")
	fmt.Println("═══════════════════════════════════════════════════════════")
	fmt.Printf("Duration:        %v\n", duration.Round(time.Second))
	fmt.Printf("Moves Detected:  %d\n", movesDetected)
	fmt.Printf("Samples Stored:  %d\n", collected)

	if duration.Seconds() > 0 {
		fmt.Printf("Avg Rate:        %.2f samples/sec\n", float64(collected)/duration.Seconds())
	}

	fmt.Println()

	displayStats(store)

	fmt.Println()
	fmt.Println("═══════════════════════════════════════════════════════════")
	fmt.Println("  Next Steps:")
	fmt.Println("───────────────────────────────────────────────────────────")

	count, _ := store.CountSamples()
	if count >= 100 {
		fmt.Println("  ✓ You have enough data to start training!")
		fmt.Println()
		fmt.Println("  Train the model:")
		fmt.Println("    ./bin/observe-train -mode=train -epochs=50")
		fmt.Println()
		fmt.Println("  Or use the advanced trainer:")
		fmt.Println("    ./bin/test-training -samples=1000 -epochs=50")
	} else {
		fmt.Printf("  ⚠️  Collect more data (have %d, recommend 100+)\n", count)
		fmt.Println()
		fmt.Println("  Continue collecting:")
		fmt.Println("    ./bin/collect-real -samples=100")
	}

	fmt.Println("═══════════════════════════════════════════════════════════")
}

// detectMove detects which move was made between two board states
// Returns move index (from*64 + to) in range 0-4095
func detectMove(oldState, newState []float64) (int, error) {
	if len(oldState) != 64 || len(newState) != 64 {
		return 0, fmt.Errorf("invalid state dimensions")
	}

	// Find squares that changed
	var changedSquares []int
	for i := 0; i < 64; i++ {
		// Use threshold to detect meaningful changes
		diff := abs(oldState[i] - newState[i])
		if diff > 0.1 { // Threshold for piece movement
			changedSquares = append(changedSquares, i)
		}
	}

	// Typical move: 2 squares change (from and to)
	// En passant or castling: 3-4 squares
	if len(changedSquares) < 2 {
		return 0, fmt.Errorf("insufficient changes detected")
	}

	// For simplicity, use the first two changed squares
	// More sophisticated logic could detect castling, en passant, etc.
	fromSquare := changedSquares[0]
	toSquare := changedSquares[1]

	// Determine which square had the piece removed (from) and added (to)
	if newState[fromSquare] >= oldState[fromSquare] {
		// toSquare was emptied, swap them
		fromSquare, toSquare = toSquare, fromSquare
	}
	// Otherwise fromSquare was emptied, no swap needed

	move := fromSquare*64 + toSquare
	return move, nil
}

// squareToNotation converts square index (0-63) to chess notation (a1-h8)
func squareToNotation(square int) string {
	if square < 0 || square >= 64 {
		return "??"
	}

	file := square % 8 // 0-7 (a-h)
	rank := square / 8 // 0-7 (1-8)
	fileChar := 'a' + rune(file)
	rankChar := '1' + rune(rank)

	return string([]rune{fileChar, rankChar})
}

// abs returns absolute value of float64
func abs(x float64) float64 {
	if x < 0 {
		return -x
	}
	return x
}

// displayStats shows current store statistics
func displayStats(store *storage.ObservationStore) {
	stats, err := store.GetStats()
	if err != nil {
		fmt.Fprintf(os.Stderr, "❌ Failed to get stats: %v\n", err)
		return
	}

	fmt.Println("Storage Statistics:")
	fmt.Println("───────────────────────────────────────────────────────────")
	fmt.Printf("  Total Samples:  %d\n", stats.TotalSamples)
	fmt.Printf("  Actual Samples: %d\n", stats.ActualSamples)
	fmt.Printf("  Max Capacity:   %d\n", stats.MaxSize)

	if stats.ActualSamples > 0 {
		utilization := float64(stats.ActualSamples) / float64(stats.MaxSize) * 100
		fmt.Printf("  Utilization:    %.1f%%\n", utilization)
	}

	if stats.IsWrapped {
		fmt.Println("  Buffer Status:  WRAPPED (oldest data overwritten)")
	} else {
		fmt.Println("  Buffer Status:  OK")
	}
	fmt.Println("───────────────────────────────────────────────────────────")
}
