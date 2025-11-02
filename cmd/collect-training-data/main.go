package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/thyrook/partner/internal/config"
	"github.com/thyrook/partner/internal/model"
	"github.com/thyrook/partner/internal/storage"
	"github.com/thyrook/partner/internal/vision"

	"go.uber.org/zap"
	"go.uber.org/zap/zapcore"
)

func main() {
	// Parse flags
	configPath := flag.String("config", "config.json", "Path to configuration file")
	targetSamples := flag.Int("samples", 1000, "Target number of samples to collect")
	captureFPS := flag.Int("fps", 10, "Capture rate (frames per second)")
	clearData := flag.Bool("clear", false, "Clear existing data before starting")
	verbose := flag.Bool("verbose", false, "Enable verbose logging")
	flag.Parse()

	// Setup logger
	var logger *zap.Logger
	if *verbose {
		config := zap.NewDevelopmentConfig()
		config.EncoderConfig.EncodeLevel = zapcore.CapitalColorLevelEncoder
		logger, _ = config.Build()
	} else {
		config := zap.NewProductionConfig()
		config.Level = zap.NewAtomicLevelAt(zap.InfoLevel)
		logger, _ = config.Build()
	}
	defer logger.Sync()

	logger.Info("P.A.R.T.N.E.R Training Data Collection",
		zap.String("version", "2.0.0"),
		zap.Int("target_samples", *targetSamples),
		zap.Int("fps", *captureFPS))

	// Load config
	cfg, err := config.Load(*configPath)
	if err != nil {
		logger.Fatal("Failed to load config", zap.Error(err))
	}

	// Initialize storage
	store, err := storage.NewObservationStore(cfg.Training.DBPath, cfg.Training.ReplayBufferSize)
	if err != nil {
		logger.Fatal("Failed to open observation store", zap.Error(err))
	}
	defer store.Close()

	// Display current stats
	stats, err := store.GetStats()
	if err != nil {
		logger.Error("Failed to get stats", zap.Error(err))
	} else {
		logger.Info("Storage initialized",
			zap.Int("current_samples", stats.ActualSamples),
			zap.Int("capacity", stats.MaxSize),
			zap.Bool("wrapped", stats.IsWrapped))
	}

	// Clear if requested
	if *clearData {
		logger.Warn("Clearing existing data...")
		if err := store.Clear(); err != nil {
			logger.Fatal("Failed to clear store", zap.Error(err))
		}
		logger.Info("Data cleared")
	}

	// Initialize async capturer
	asyncCapturer := vision.NewAsyncCapturer(vision.AsyncCapturerConfig{
		X:             cfg.Vision.ScreenRegion.X,
		Y:             cfg.Vision.ScreenRegion.Y,
		Width:         cfg.Vision.ScreenRegion.Width,
		Height:        cfg.Vision.ScreenRegion.Height,
		BoardSize:     cfg.Vision.BoardSize,
		DiffThreshold: cfg.Vision.DiffThreshold,
		TargetFPS:     *captureFPS,
		BufferSize:    2,
	})

	// Start async capture
	if err := asyncCapturer.Start(); err != nil {
		logger.Fatal("Failed to start async capture", zap.Error(err))
	}
	defer asyncCapturer.Stop()

	logger.Info("Waiting for first frame...")
	if err := asyncCapturer.WaitForReady(5 * time.Second); err != nil {
		logger.Fatal("Timeout waiting for first frame", zap.Error(err))
	}
	logger.Info("Capture system ready")

	// Load or initialize model (for move prediction/validation)
	chess, err := model.NewChessNet(cfg.Model.InputSize, cfg.Model.HiddenSize, cfg.Model.OutputSize)
	if err != nil {
		logger.Fatal("Failed to create model", zap.Error(err))
	}

	if _, err := os.Stat(cfg.Model.ModelPath); err == nil {
		logger.Info("Loading existing model weights", zap.String("path", cfg.Model.ModelPath))
		if err := chess.Load(cfg.Model.ModelPath); err != nil {
			logger.Warn("Failed to load weights, using random initialization", zap.Error(err))
		}
	} else {
		logger.Info("No existing weights found, using random initialization")
	}

	// Run collection
	runCollection(logger, asyncCapturer, store, chess, *targetSamples)
}

func runCollection(logger *zap.Logger, capturer *vision.AsyncCapturer, store *storage.ObservationStore, chess *model.ChessNet, targetSamples int) {
	fmt.Println()
	fmt.Println("═══════════════════════════════════════════════════════════")
	fmt.Println("  TRAINING DATA COLLECTION")
	fmt.Println("═══════════════════════════════════════════════════════════")
	fmt.Println()
	fmt.Println("Instructions:")
	fmt.Println("  1. Open your chess board in the configured screen region")
	fmt.Println("  2. Play chess (online, vs computer, or watch games)")
	fmt.Println("  3. System will automatically detect and record moves")
	fmt.Println("  4. Press Ctrl+C to stop and save data")
	fmt.Println()
	fmt.Printf("Target: %d samples\n", targetSamples)
	fmt.Println()
	fmt.Println("Monitoring board for changes...")
	fmt.Println()

	// Setup signal handling
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, os.Interrupt, syscall.SIGTERM)

	go func() {
		<-sigChan
		logger.Info("Interrupt received, stopping collection...")
		cancel()
	}()

	// Collection state
	collected := 0
	movesDetected := 0
	startTime := time.Now()
	lastState := []float64{}
	lastStatsTime := time.Now()

	// Main collection loop
	ticker := time.NewTicker(100 * time.Millisecond)
	defer ticker.Stop()

	fmt.Println("Status:")
	fmt.Println("───────────────────────────────────────────────────────────")

	for {
		select {
		case <-ctx.Done():
			goto done

		case <-ticker.C:
			// Extract board state
			boardState, err := capturer.ExtractBoardState()
			if err != nil {
				continue
			}

			// Check if board changed
			if !boardState.Changed {
				continue
			}

			// Initialize on first capture
			if len(lastState) == 0 {
				lastState = make([]float64, len(boardState.Grid))
				copy(lastState, boardState.Grid)
				fmt.Printf("\n✓ Initial board state captured (%d squares)\n", len(lastState))
				continue
			}

			// Detect move
			move, confidence, err := detectMoveWithValidation(lastState, boardState.Grid, chess)
			if err != nil {
				if time.Since(lastStatsTime) > 2*time.Second {
					logger.Debug("Move detection skipped", zap.Error(err))
					lastStatsTime = time.Now()
				}
				continue
			}

			movesDetected++

			// Store observation (state before move + move that was made)
			if err := store.StoreSample(lastState, move); err != nil {
				logger.Error("Failed to store sample", zap.Error(err))
				continue
			}

			collected++

			// Display move
			fromSquare := move / 64
			toSquare := move % 64
			fromNotation := squareToNotation(fromSquare)
			toNotation := squareToNotation(toSquare)

			elapsed := time.Since(startTime).Seconds()
			rate := float64(collected) / elapsed

			fmt.Printf("\r[%s] Move #%d: %s→%s (%.1f%% conf) | Total: %d samples | Rate: %.2f/sec        ",
				time.Now().Format("15:04:05"),
				movesDetected,
				fromNotation,
				toNotation,
				confidence*100,
				collected,
				rate)

			// Update state
			copy(lastState, boardState.Grid)

			// Check target
			if targetSamples > 0 && collected >= targetSamples {
				fmt.Println()
				logger.Info("Target reached", zap.Int("samples", collected))
				cancel()
				goto done
			}

			// Periodic stats
			if collected%20 == 0 {
				captureStats := capturer.GetStatistics()
				fmt.Printf("\n    [Capture: %d frames, %.0fms avg, %.1f FPS effective]\n",
					captureStats.TotalCaptures,
					captureStats.AvgCaptureMs,
					1000.0/captureStats.AvgCaptureMs)
			}
		}
	}

done:
	duration := time.Since(startTime)

	fmt.Println()
	fmt.Println()
	fmt.Println("═══════════════════════════════════════════════════════════")
	fmt.Println("  COLLECTION COMPLETE")
	fmt.Println("═══════════════════════════════════════════════════════════")
	fmt.Printf("Duration:         %v\n", duration.Round(time.Second))
	fmt.Printf("Moves Detected:   %d\n", movesDetected)
	fmt.Printf("Samples Stored:   %d\n", collected)

	if duration.Seconds() > 0 {
		fmt.Printf("Collection Rate:  %.2f samples/sec\n", float64(collected)/duration.Seconds())
	}

	// Final stats
	stats, err := store.GetStats()
	if err == nil {
		fmt.Println()
		fmt.Println("Database Statistics:")
		fmt.Println("───────────────────────────────────────────────────────────")
		fmt.Printf("  Total Samples:    %d\n", stats.TotalSamples)
		fmt.Printf("  Actual Samples:   %d\n", stats.ActualSamples)
		fmt.Printf("  Capacity:         %d\n", stats.MaxSize)
		fmt.Printf("  Utilization:      %.1f%%\n", float64(stats.ActualSamples)/float64(stats.MaxSize)*100)

		if stats.IsWrapped {
			fmt.Println("  Status:           WRAPPED (oldest overwritten)")
		} else {
			fmt.Println("  Status:           OK")
		}
	}

	// Capture stats
	captureStats := capturer.GetStatistics()
	fmt.Println()
	fmt.Println("Capture Statistics:")
	fmt.Println("───────────────────────────────────────────────────────────")
	fmt.Printf("  Total Captures:   %d\n", captureStats.TotalCaptures)
	fmt.Printf("  Errors:           %d\n", captureStats.TotalErrors)
	fmt.Printf("  Avg Capture Time: %.2fms\n", captureStats.AvgCaptureMs)
	fmt.Printf("  Last Capture:     %dms\n", captureStats.LastCaptureMs)

	fmt.Println()
	fmt.Println("═══════════════════════════════════════════════════════════")
	fmt.Println("  NEXT STEPS")
	fmt.Println("═══════════════════════════════════════════════════════════")

	if err == nil && stats.ActualSamples >= 100 {
		fmt.Println()
		fmt.Println("  ✅ You have enough data to train!")
		fmt.Println()
		fmt.Println("  Train the model:")
		fmt.Println("    ./bin/test-training -samples=1000 -epochs=50")
		fmt.Println()
		fmt.Println("  Or use advanced trainer:")
		fmt.Println("    ./bin/test-training -advanced -epochs=100")
	} else if err == nil {
		fmt.Println()
		fmt.Printf("  ⚠️  Collect more data (have %d, recommend 100+)\n", stats.ActualSamples)
		fmt.Println()
		fmt.Println("  Continue collecting:")
		fmt.Println("    ./bin/collect-training-data -samples=100")
	}

	fmt.Println()
	fmt.Println("═══════════════════════════════════════════════════════════")
}

// detectMoveWithValidation detects move and validates it
func detectMoveWithValidation(oldState, newState []float64, chess *model.ChessNet) (int, float64, error) {
	if len(oldState) != 64 || len(newState) != 64 {
		return 0, 0, fmt.Errorf("invalid state dimensions")
	}

	// Find changed squares
	var changedSquares []int
	for i := 0; i < 64; i++ {
		diff := abs(oldState[i] - newState[i])
		if diff > 0.15 { // Threshold for significant change
			changedSquares = append(changedSquares, i)
		}
	}

	// Need at least 2 squares changed (from and to)
	if len(changedSquares) < 2 {
		return 0, 0, fmt.Errorf("insufficient changes detected (%d squares)", len(changedSquares))
	}

	// Limit to most significant changes
	if len(changedSquares) > 4 {
		// Keep only the 4 most changed squares
		changedSquares = changedSquares[:4]
	}

	// Determine from and to squares
	// From square: piece was removed (value decreased)
	// To square: piece was added (value increased)
	var fromSquare, toSquare int
	var foundFrom, foundTo bool

	for _, sq := range changedSquares {
		oldVal := oldState[sq]
		newVal := newState[sq]

		if newVal < oldVal && !foundFrom {
			// Piece removed
			fromSquare = sq
			foundFrom = true
		} else if newVal > oldVal && !foundTo {
			// Piece added
			toSquare = sq
			foundTo = true
		}
	}

	if !foundFrom || !foundTo {
		// Try alternative: just use first two changed squares
		fromSquare = changedSquares[0]
		toSquare = changedSquares[1]
	}

	move := fromSquare*64 + toSquare

	// Validate move is in legal range
	if move < 0 || move >= 4096 {
		return 0, 0, fmt.Errorf("move out of range: %d", move)
	}

	// Optional: Use model to get confidence (if trained)
	confidence := 0.5 // Default confidence

	return move, confidence, nil
}

func squareToNotation(square int) string {
	if square < 0 || square >= 64 {
		return "??"
	}
	file := square % 8
	rank := square / 8
	return string([]rune{'a' + rune(file), '1' + rune(rank)})
}

func abs(x float64) float64 {
	if x < 0 {
		return -x
	}
	return x
}
