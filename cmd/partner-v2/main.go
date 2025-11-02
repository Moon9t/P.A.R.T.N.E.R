package main

import (
	"flag"
	"fmt"
	"net/http"
	_ "net/http/pprof"
	"os"
	"os/signal"
	"runtime"
	"syscall"
	"time"

	"go.uber.org/zap"

	"github.com/thyrook/partner/internal/config"
	"github.com/thyrook/partner/internal/decision"
	"github.com/thyrook/partner/internal/iface"
	"github.com/thyrook/partner/internal/model"
	"github.com/thyrook/partner/internal/storage"
	"github.com/thyrook/partner/internal/training"
	"github.com/thyrook/partner/internal/vision"
)

// Version information
const (
	Version   = "1.0.0"
	Phase     = "Phase 5"
	BuildDate = "2025-11-02"
)

func main() {
	// Parse command line flags
	var (
		configPath  = flag.String("config", "config.json", "Path to configuration file")
		mode        = flag.String("mode", "assist", "Operation mode: assist, train, collect, stats")
		profiling   = flag.Bool("profile", false, "Enable pprof profiling on :6060")
		verbose     = flag.Bool("verbose", false, "Enable verbose logging")
		noColor     = flag.Bool("no-color", false, "Disable colored output")
		enableTTS   = flag.Bool("tts", false, "Enable text-to-speech")
		trainEpochs = flag.Int("epochs", 50, "Training epochs")
		collectNum  = flag.Int("collect", 100, "Number of samples to collect")
	)
	flag.Parse()

	// Enable profiling if requested
	if *profiling {
		go func() {
			fmt.Println("🔍 Profiling enabled on http://localhost:6060/debug/pprof")
			if err := http.ListenAndServe(":6060", nil); err != nil {
				fmt.Printf("⚠️  Profiling server failed: %v\n", err)
			}
		}()
	}

	// Load configuration
	cfg, err := config.Load(*configPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "❌ Failed to load configuration: %v\n", err)
		os.Exit(1)
	}

	// Override log level if verbose
	if *verbose {
		cfg.Interface.LogLevel = "debug"
	}

	// Initialize logger
	logger, err := iface.NewLogger(cfg.Interface.LogPath, cfg.Interface.LogLevel)
	if err != nil {
		fmt.Fprintf(os.Stderr, "❌ Failed to initialize logger: %v\n", err)
		os.Exit(1)
	}
	defer logger.Sync()

	// Create enhanced CLI
	cli := iface.NewEnhancedCLI(logger, !*noColor)
	cli.PrintBanner()

	// Log startup
	logger.Info("P.A.R.T.N.E.R starting",
		zap.String("version", Version),
		zap.String("phase", Phase),
		zap.String("mode", *mode),
		zap.String("build_date", BuildDate),
		zap.String("go_version", runtime.Version()),
	)

	// Ensure data directory exists
	if err := os.MkdirAll("data", 0755); err != nil {
		logger.Error("Failed to create data directory", zap.Error(err))
		os.Exit(1)
	}

	// Setup signal handling for graceful shutdown
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, os.Interrupt, syscall.SIGTERM)

	// Run the appropriate mode
	switch *mode {
	case "assist":
		runAssistMode(cfg, logger, cli, *enableTTS, sigChan)
	case "train":
		runTrainMode(cfg, logger, cli, *trainEpochs, sigChan)
	case "collect":
		runCollectMode(cfg, logger, cli, *collectNum, sigChan)
	case "stats":
		runStatsMode(cfg, logger, cli)
	default:
		fmt.Fprintf(os.Stderr, "❌ Unknown mode: %s\n", *mode)
		fmt.Fprintf(os.Stderr, "Available modes: assist, train, collect, stats\n")
		os.Exit(1)
	}

	logger.Info("P.A.R.T.N.E.R shutting down gracefully")
	cli.PrintStatus("Goodbye! Chess awaits your next move.", "success")
}

// runAssistMode runs the real-time chess assistance mode
func runAssistMode(
	cfg *config.Config,
	logger *iface.Logger,
	cli *iface.EnhancedCLI,
	enableTTS bool,
	sigChan chan os.Signal,
) {
	cli.PrintStatus("Initializing Chess Assistance Mode", "info")

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
	cli.PrintStatus("Validating screen capture system...", "info")
	if err := capturer.ValidateCapture(); err != nil {
		cli.PrintError(err, "Screen capture validation failed")
		fmt.Println("\nPlease ensure:")
		fmt.Println("  1. Chess board is visible on screen")
		fmt.Println("  2. config.json has correct screen region")
		fmt.Println("  3. Screen capture permissions are granted")
		os.Exit(1)
	}
	cli.PrintStatus("Screen capture system ready", "success")

	// Initialize neural network
	cli.PrintStatus("Loading neural network...", "info")
	net, err := model.NewChessNet(cfg.Model.InputSize, cfg.Model.HiddenSize, cfg.Model.OutputSize)
	if err != nil {
		cli.PrintError(err, "Failed to create neural network")
		os.Exit(1)
	}
	defer net.Close()

	// Load model if available
	if model.ModelExists(cfg.Model.ModelPath) {
		cli.PrintStatus("Loading trained model weights...", "info")
		if err := net.Load(cfg.Model.ModelPath); err != nil {
			cli.PrintStatus("Failed to load model, using fresh weights", "warning")
			logger.Warn("Model load failed", zap.Error(err))
		} else {
			cli.PrintStatus("Model loaded successfully", "success")
		}
	} else {
		cli.PrintStatus("No trained model found, using fresh weights", "warning")
		fmt.Println("  Run in 'train' mode first for better results")
	}

	// Initialize decision engine with zap logger
	zapLogger := logger.GetZapLogger()
	engine := decision.NewDecisionEngine(
		net,
		capturer,
		cfg.Interface.ConfidenceThreshold,
		5, // top 5 moves
		zapLogger,
	)

	// Initialize decision history
	history := decision.NewDecisionHistory(100)

	// Initialize voice feedback
	voice := iface.NewEnhancedVoice(enableTTS, zapLogger)
	if voice.IsEnabled() {
		cli.PrintStatus(fmt.Sprintf("Voice feedback enabled (engine: %s)", voice.GetEngine()), "success")
	}

	// Start assistance loop
	cli.PrintStatus("Starting real-time assistance...", "success")
	fmt.Println()
	fmt.Println("♟️  Watching chess board for move suggestions")
	fmt.Println("⌨️  Press Ctrl+C to stop")
	fmt.Println()

	ticker := time.NewTicker(2 * time.Second)
	defer ticker.Stop()

	decisionCount := 0
	lastBoardHash := ""

	for {
		select {
		case <-sigChan:
			fmt.Println()
			fmt.Println()
			cli.PrintStatus("Received shutdown signal", "info")
			goto shutdown

		case <-ticker.C:
			// Make decision
			dec, err := engine.MakeDecision()
			if err != nil {
				// Only log errors, don't display to avoid spam
				logger.Debug("Decision failed", zap.Error(err))
				continue
			}

			// Simple deduplication using board state hash
			boardHash := fmt.Sprintf("%v", dec.BoardState[0:8])
			if boardHash == lastBoardHash {
				continue
			}
			lastBoardHash = boardHash

			decisionCount++

			// Add to history
			history.Add(dec)

			// Display decision
			cli.PrintDecision(dec)

			// Announce via voice
			if voice.IsEnabled() {
				voice.AnnounceDecision(dec)
			}

			// Periodic stats (every 10 decisions)
			if decisionCount%10 == 0 {
				stats := engine.GetStatistics()
				histStats := history.GetStats()
				cli.PrintStatistics(stats, histStats)

				if voice.IsEnabled() {
					voice.AnnounceStatistics(stats)
				}
			}
		}
	}

shutdown:
	// Display final statistics
	fmt.Println()
	cli.PrintStatus("Generating final statistics...", "info")
	stats := engine.GetStatistics()
	histStats := history.GetStats()
	cli.PrintStatistics(stats, histStats)

	logger.Info("Assist mode completed",
		zap.Int("total_decisions", decisionCount),
		zap.Float64("avg_inference_ms", stats.AvgInferenceMs),
	)
}

// runTrainMode runs the training mode
func runTrainMode(
	cfg *config.Config,
	logger *iface.Logger,
	cli *iface.EnhancedCLI,
	epochs int,
	sigChan chan os.Signal,
) {
	cli.PrintStatus("Initializing Training Mode", "info")

	// Initialize storage
	cli.PrintStatus("Opening observation database...", "info")
	store, err := storage.NewObservationStore(cfg.Training.DBPath, cfg.Training.ReplayBufferSize)
	if err != nil {
		cli.PrintError(err, "Failed to open observation store")
		os.Exit(1)
	}
	defer store.Close()

	// Check sample count
	count, err := store.CountSamples()
	if err != nil {
		cli.PrintError(err, "Failed to count samples")
		os.Exit(1)
	}

	cli.PrintStatus(fmt.Sprintf("Found %d training samples", count), "info")

	if count < uint64(cfg.Training.MinSamplesBeforeTraining) {
		cli.PrintStatus(fmt.Sprintf("Insufficient samples (need %d)", cfg.Training.MinSamplesBeforeTraining), "error")
		fmt.Println("\nCollect more data first:")
		fmt.Printf("  %s -mode=collect -collect=%d\n",
			os.Args[0], cfg.Training.MinSamplesBeforeTraining)
		os.Exit(1)
	}

	// Initialize model
	cli.PrintStatus("Initializing neural network...", "info")
	net, err := model.NewChessNet(cfg.Model.InputSize, cfg.Model.HiddenSize, cfg.Model.OutputSize)
	if err != nil {
		cli.PrintError(err, "Failed to create model")
		os.Exit(1)
	}
	defer net.Close()

	// Load existing weights if available
	if model.ModelExists(cfg.Model.ModelPath) {
		cli.PrintStatus("Loading existing model...", "info")
		if err := net.Load(cfg.Model.ModelPath); err != nil {
			cli.PrintStatus("Failed to load, starting fresh", "warning")
		} else {
			cli.PrintStatus("Model loaded", "success")
		}
	}

	// Training configuration
	trainConfig := &training.TrainingConfig{
		Epochs:       epochs,
		BatchSize:    cfg.Model.BatchSize,
		LearningRate: cfg.Model.LearningRate,
		Verbose:      false,
	}

	// Create trainer
	trainer, err := training.NewAdvancedTrainer(net, store, trainConfig)
	if err != nil {
		cli.PrintError(err, "Failed to create trainer")
		os.Exit(1)
	}

	cli.PrintStatus(fmt.Sprintf("Training for %d epochs...", epochs), "info")
	fmt.Println()

	stopped := false
	go func() {
		<-sigChan
		stopped = true
		cli.PrintStatus("Training interrupted, saving progress...", "warning")
	}()

	// Progress callback
	lastUpdate := time.Now()
	progressCallback := func(metrics *training.EpochMetrics) {
		if stopped {
			return
		}

		// Update progress
		cli.PrintProgress(metrics.Epoch, epochs, "Training")

		// Detailed update every 10 epochs or if 1 second passed
		if metrics.Epoch%10 == 0 || time.Since(lastUpdate) > time.Second {
			logger.Info("Training progress",
				zap.Int("epoch", metrics.Epoch),
				zap.Float64("loss", metrics.Loss),
				zap.Int("samples", metrics.SamplesUsed),
				zap.Duration("duration", metrics.Duration),
			)
			lastUpdate = time.Now()
		}

		// Save checkpoint every N epochs
		if metrics.Epoch > 0 && metrics.Epoch%cfg.Training.SaveInterval == 0 {
			if err := net.Save(cfg.Model.ModelPath); err != nil {
				logger.Error("Checkpoint save failed", zap.Error(err))
			} else {
				cli.PrintStatus(fmt.Sprintf("Checkpoint saved (epoch %d)", metrics.Epoch), "success")
			}
		}
	}

	// Train
	startTime := time.Now()
	result, err := trainer.Train(trainConfig, progressCallback)
	duration := time.Since(startTime)

	if err != nil {
		cli.PrintError(err, "Training failed")
		os.Exit(1)
	}

	fmt.Println()
	cli.PrintStatus(fmt.Sprintf("Training completed in %v", duration.Round(time.Second)), "success")

	// Display metrics
	if len(result.Metrics) > 0 {
		lastMetrics := result.Metrics[len(result.Metrics)-1]
		fmt.Println()
		fmt.Printf("Final Loss:       %.6f\n", lastMetrics.Loss)
		fmt.Printf("Total Samples:    %d\n", lastMetrics.SamplesUsed*epochs)
		fmt.Printf("Valid Batches:    %d/%d\n", lastMetrics.BatchesValid, lastMetrics.BatchesTotal)
		if lastMetrics.BatchesValid > 0 {
			avgBatchTime := lastMetrics.Duration / time.Duration(lastMetrics.BatchesValid)
			fmt.Printf("Avg Batch Time:   %v\n", avgBatchTime.Round(time.Millisecond))
		}
	}

	// Save final model
	cli.PrintStatus("Saving final model...", "info")
	if err := net.Save(cfg.Model.ModelPath); err != nil {
		cli.PrintError(err, "Failed to save model")
		os.Exit(1)
	}

	cli.PrintStatus("Model saved successfully", "success")

	finalLoss := 0.0
	if len(result.Metrics) > 0 {
		finalLoss = result.Metrics[len(result.Metrics)-1].Loss
	}

	logger.Info("Training completed",
		zap.Int("epochs", epochs),
		zap.Float64("final_loss", finalLoss),
		zap.Duration("duration", duration),
	)
}

// runCollectMode runs observation collection mode
func runCollectMode(
	cfg *config.Config,
	logger *iface.Logger,
	cli *iface.EnhancedCLI,
	numSamples int,
	sigChan chan os.Signal,
) {
	cli.PrintStatus("Initializing Collection Mode", "info")

	// Initialize storage
	store, err := storage.NewObservationStore(cfg.Training.DBPath, cfg.Training.ReplayBufferSize)
	if err != nil {
		cli.PrintError(err, "Failed to open observation store")
		os.Exit(1)
	}
	defer store.Close()

	// Initialize vision
	capturer := vision.NewCapturer(
		cfg.Vision.ScreenRegion.X,
		cfg.Vision.ScreenRegion.Y,
		cfg.Vision.ScreenRegion.Width,
		cfg.Vision.ScreenRegion.Height,
		cfg.Vision.BoardSize,
		cfg.Vision.DiffThreshold,
	)
	defer capturer.Close()

	// Validate capture
	if err := capturer.ValidateCapture(); err != nil {
		cli.PrintError(err, "Capture validation failed")
		os.Exit(1)
	}

	cli.PrintStatus(fmt.Sprintf("Collecting %d samples...", numSamples), "info")
	cli.PrintStatus("Play chess and moves will be recorded automatically", "info")
	fmt.Println()

	collected := 0
	ticker := time.NewTicker(500 * time.Millisecond)
	defer ticker.Stop()

	for collected < numSamples {
		select {
		case <-sigChan:
			cli.PrintStatus("Collection interrupted", "warning")
			goto done

		case <-ticker.C:
			boardState, err := capturer.ExtractBoardState()
			if err != nil {
				continue
			}

			if !boardState.Changed {
				continue
			}

			// Simple move detection (placeholder)
			move := 0 // In real implementation, detect actual move

			if err := store.StoreSample(boardState.Grid, move); err != nil {
				logger.Error("Failed to store sample", zap.Error(err))
				continue
			}

			collected++
			cli.PrintProgress(collected, numSamples, "Collecting")
		}
	}

done:
	fmt.Println()
	cli.PrintStatus(fmt.Sprintf("Collected %d samples", collected), "success")

	logger.Info("Collection completed",
		zap.Int("samples_collected", collected),
	)
}

// runStatsMode displays system statistics
func runStatsMode(
	cfg *config.Config,
	logger *iface.Logger,
	cli *iface.EnhancedCLI,
) {
	cli.PrintStatus("System Statistics", "info")
	fmt.Println()

	// Storage stats
	store, err := storage.NewObservationStore(cfg.Training.DBPath, cfg.Training.ReplayBufferSize)
	if err != nil {
		cli.PrintError(err, "Failed to open storage")
		return
	}
	defer store.Close()

	stats, err := store.GetStats()
	if err != nil {
		cli.PrintError(err, "Failed to get storage stats")
		return
	}

	fmt.Println("📊 Storage Statistics")
	fmt.Println("─────────────────────")
	fmt.Printf("Total Samples:  %d\n", stats.TotalSamples)
	fmt.Printf("Actual Samples: %d\n", stats.ActualSamples)
	fmt.Printf("Max Capacity:   %d\n", stats.MaxSize)
	fmt.Printf("Utilization:    %.1f%%\n", float64(stats.ActualSamples)/float64(stats.MaxSize)*100)
	fmt.Printf("Wrapped:        %v\n", stats.IsWrapped)
	fmt.Println()

	// Model stats
	if model.ModelExists(cfg.Model.ModelPath) {
		info, err := os.Stat(cfg.Model.ModelPath)
		if err == nil {
			fmt.Println("🧠 Model Information")
			fmt.Println("──────────────────")
			fmt.Printf("Model Path:     %s\n", cfg.Model.ModelPath)
			fmt.Printf("Model Size:     %.2f MB\n", float64(info.Size())/1024/1024)
			fmt.Printf("Last Modified:  %s\n", info.ModTime().Format("2006-01-02 15:04:05"))
			fmt.Println()
		}
	} else {
		fmt.Println("🧠 Model Information")
		fmt.Println("──────────────────")
		fmt.Println("No trained model found")
		fmt.Println()
	}

	logger.Info("Statistics displayed")
}
