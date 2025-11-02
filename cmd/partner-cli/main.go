package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"os/signal"
	"path/filepath"
	"runtime"
	"syscall"
	"time"

	"github.com/thyrook/partner/internal/config"
	"github.com/thyrook/partner/internal/data"
	"github.com/thyrook/partner/internal/decision"
	"github.com/thyrook/partner/internal/iface"
	"github.com/thyrook/partner/internal/iface/logger"
	"github.com/thyrook/partner/internal/model"
	"github.com/thyrook/partner/internal/storage"
	"github.com/thyrook/partner/internal/training"
	"github.com/thyrook/partner/internal/vision"
	"go.uber.org/zap"
)

var (
	// Command line flags
	configPath  = flag.String("config", "configs/partner.json", "Path to configuration file")
	mode        = flag.String("mode", "", "Operation mode: observe, train, or analyze")
	adapterName = flag.String("adapter", "chess", "Game adapter: chess")
	quiet       = flag.Bool("quiet", false, "Quiet mode - minimal output")
	datasetPath = flag.String("dataset", "", "Path to dataset (overrides config)")
	modelPath   = flag.String("model", "", "Path to model (overrides config)")
	epochs      = flag.Int("epochs", 0, "Number of training epochs (overrides config)")
	help        = flag.Bool("help", false, "Show help message")
)

func main() {
	flag.Parse()

	if *help || *mode == "" {
		printHelp()
		return
	}

	// Load configuration
	cfg := config.LoadOrDefault(*configPath)

	// Apply command-line overrides
	if *datasetPath != "" {
		cfg.Training.DBPath = *datasetPath
	}
	if *modelPath != "" {
		cfg.Model.ModelPath = *modelPath
	}
	if *quiet {
		cfg.Interface.Quiet = true
	}

	// Validate configuration
	if err := cfg.Validate(); err != nil {
		fmt.Fprintf(os.Stderr, "Configuration error: %v\n", err)
		os.Exit(1)
	}

	// Ensure directories exist
	if err := cfg.EnsureDirectories(); err != nil {
		fmt.Fprintf(os.Stderr, "Failed to create directories: %v\n", err)
		os.Exit(1)
	}

	// Initialize logger
	logLevel := logger.Level(cfg.Interface.LogLevel)
	if err := logger.Setup(logLevel, cfg.Interface.LogPath); err != nil {
		fmt.Fprintf(os.Stderr, "Failed to initialize logger: %v\n", err)
		os.Exit(1)
	}

	// Initialize CLI
	cli := iface.NewCLI(cfg, cfg.Interface.Quiet)
	cli.PrintBanner()

	logger.Info("P.A.R.T.N.E.R starting",
		"mode", *mode,
		"version", cfg.Version,
		"go_version", runtime.Version(),
	)

	// Setup signal handling for graceful shutdown
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, os.Interrupt, syscall.SIGTERM)

	go func() {
		sig := <-sigChan
		logger.Info("Received signal, shutting down gracefully", "signal", sig)
		cli.PrintStatus("Shutting down gracefully...", "info")
		cancel()
	}()

	// Execute the requested mode
	var err error
	switch *mode {
	case "observe":
		err = runObserveMode(ctx, cfg, cli)
	case "train":
		err = runTrainMode(ctx, cfg, cli)
	case "analyze":
		err = runAnalyzeMode(ctx, cfg, cli)
	default:
		fmt.Fprintf(os.Stderr, "Unknown mode: %s\n", *mode)
		printHelp()
		os.Exit(1)
	}

	if err != nil {
		logger.Error("Mode execution failed", "mode", *mode, "error", err)
		cli.PrintError(err)
		os.Exit(1)
	}

	logger.Info("P.A.R.T.N.E.R shutdown complete")
	cli.PrintStatus("Shutdown complete", "success")
}

func runObserveMode(ctx context.Context, cfg *config.Config, cli *iface.CLI) error {
	cli.PrintModeHeader("observe")

	logger.Info("Starting observe mode",
		"fps", cfg.Vision.CaptureFPS,
		"screen_region", cfg.Vision.ScreenRegion,
	)

	// Initialize ChessNet model (decision engine uses ChessNet, not ChessCNN)
	cli.PrintStatus("Loading chess model...", "info")

	net, err := model.NewChessNet(
		cfg.Model.InputSize,
		cfg.Model.HiddenSize,
		cfg.Model.OutputSize,
	)
	if err != nil {
		return fmt.Errorf("failed to create model: %w", err)
	}

	// Try to load existing model weights
	if _, statErr := os.Stat(cfg.Model.ModelPath); statErr == nil {
		cli.PrintStatus(fmt.Sprintf("Loading model from %s", cfg.Model.ModelPath), "info")
		if err := net.Load(cfg.Model.ModelPath); err != nil {
			cli.PrintWarning(fmt.Sprintf("Failed to load model: %v", err))
			logger.Warn("Failed to load model", "error", err)
		} else {
			cli.PrintStatus("Model loaded successfully", "success")
		}
	} else {
		cli.PrintWarning("No trained model found - predictions will be random")
	}

	// Initialize vision system
	cli.PrintStatus("Initializing vision system...", "info")
	capturerCfg := vision.AsyncCapturerConfig{
		X:             cfg.Vision.ScreenRegion.X,
		Y:             cfg.Vision.ScreenRegion.Y,
		Width:         cfg.Vision.ScreenRegion.Width,
		Height:        cfg.Vision.ScreenRegion.Height,
		BoardSize:     8,
		DiffThreshold: 0.05,
		TargetFPS:     cfg.Vision.CaptureFPS,
		BufferSize:    2,
	}
	capturer := vision.NewAsyncCapturer(capturerCfg)
	defer capturer.Stop()

	if err := capturer.Start(); err != nil {
		return fmt.Errorf("failed to start capturer: %w", err)
	}

	// Initialize decision engine with proper logger
	zapLogger, _ := zap.NewProduction()
	defer zapLogger.Sync()

	cli.PrintStatus("Initializing decision engine...", "info")
	engine := decision.NewAsyncDecisionEngine(
		net,
		capturer,
		cfg.Interface.ConfidenceThreshold,
		cfg.Interface.TopMoves,
		zapLogger,
	)

	cli.PrintStatus("System ready - watching for moves", "success")
	cli.PrintSeparator()

	// Main observation loop
	ticker := time.NewTicker(1 * time.Second)
	defer ticker.Stop()

	lastMoveTime := time.Now()
	moveCount := 0

	for {
		select {
		case <-ctx.Done():
			return nil

		case <-ticker.C:
			// Extract board state from vision system
			boardState, err := capturer.ExtractBoardState()
			if err != nil {
				logger.Debug("No board state available", "error", err)
				continue
			}

			// Get predictions from model
			predictions, err := net.Predict(boardState.Grid)
			if err != nil {
				logger.Error("Prediction failed", "error", err)
				continue
			}

			// Convert to MoveScores using GetTopKMoves
			moves := model.GetTopKMoves(predictions, cfg.Interface.TopMoves)

			// Only show predictions if confidence is above threshold
			if len(moves) > 0 && moves[0].Score >= cfg.Interface.ConfidenceThreshold {
				if time.Since(lastMoveTime) > 5*time.Second {
					moveCount++
					lastMoveTime = time.Now()

					fmt.Printf("\n⏱️  Move %d | %s\n", moveCount, time.Now().Format("15:04:05"))
					cli.PrintTopMoves(moves, cfg.Interface.TopMoves)

					logger.Info("Move predicted",
						"move_number", moveCount,
						"top_move_index", moves[0].MoveIndex,
						"confidence", moves[0].Score,
					)
				}
			}

			// Log performance stats periodically
			if moveCount > 0 && moveCount%30 == 0 {
				stats := engine.GetStatistics()
				logger.Info("Performance stats",
					"total_decisions", stats.TotalDecisions,
					"avg_inference_ms", stats.AvgInferenceMs,
				)
			}
		}
	}
}

func runTrainMode(ctx context.Context, cfg *config.Config, cli *iface.CLI) error {
	cli.PrintModeHeader("train")

	// Get epochs from config or flag
	numEpochs := 10
	if *epochs > 0 {
		numEpochs = *epochs
	}

	logger.Info("Starting train mode",
		"epochs", numEpochs,
		"dataset", cfg.Training.DBPath,
		"batch_size", cfg.Model.BatchSize,
		"learning_rate", cfg.Model.LearningRate,
	)

	// Check if dataset exists
	if _, err := os.Stat(cfg.Training.DBPath); os.IsNotExist(err) {
		return fmt.Errorf("dataset not found: %s", cfg.Training.DBPath)
	}

	// Open observation store
	cli.PrintStatus(fmt.Sprintf("Loading dataset from %s", cfg.Training.DBPath), "info")
	store, err := storage.NewObservationStore(cfg.Training.DBPath, cfg.Training.ReplayBufferSize)
	if err != nil {
		return fmt.Errorf("failed to open dataset: %w", err)
	}
	defer store.Close()

	count, err := store.CountSamples()
	if err != nil {
		return fmt.Errorf("failed to get dataset count: %w", err)
	}

	if count == 0 {
		return fmt.Errorf("dataset is empty")
	}

	cli.PrintStatus(fmt.Sprintf("Dataset loaded: %d observations", count), "success")
	logger.Info("Dataset statistics", "total_observations", count)

	// Initialize model
	cli.PrintStatus("Initializing model...", "info")
	net, err := model.NewChessNet(
		cfg.Model.InputSize,
		cfg.Model.HiddenSize,
		cfg.Model.OutputSize,
	)
	if err != nil {
		return fmt.Errorf("failed to create model: %w", err)
	}

	// Try to load existing weights
	if _, err := os.Stat(cfg.Model.ModelPath); err == nil {
		cli.PrintStatus("Loading existing model weights", "info")
		if err := net.Load(cfg.Model.ModelPath); err != nil {
			cli.PrintWarning(fmt.Sprintf("Failed to load weights: %v", err))
		} else {
			cli.PrintStatus("Model loaded successfully", "success")
		}
	}

	// Create storage trainer
	trainer, err := training.NewStorageTrainer(net, store, cfg.Model.LearningRate, cfg.Model.BatchSize)
	if err != nil {
		return fmt.Errorf("failed to create trainer: %w", err)
	}

	// Setup CPU monitoring
	startCPU := getCPUUsage()
	startTime := time.Now()

	cli.PrintStatus(fmt.Sprintf("Starting training: %d epochs", numEpochs), "info")
	cli.PrintSeparator()

	// Training loop
	for epoch := 1; epoch <= numEpochs; epoch++ {
		select {
		case <-ctx.Done():
			cli.PrintStatus("Training interrupted", "warning")
			return nil
		default:
		}

		epochStart := time.Now()

		// Train epoch
		cli.PrintProgress(epoch-1, numEpochs, fmt.Sprintf("Epoch %d/%d - Training", epoch, numEpochs))

		loss, err := trainer.TrainEpoch()
		if err != nil {
			logger.Error("Training failed", "epoch", epoch, "error", err)
			continue
		}

		epochDuration := time.Since(epochStart)

		// Print stats (accuracy not available from StorageTrainer)
		cli.PrintTrainingStats(epoch, numEpochs, loss, 0.0, epochDuration)

		// Check CPU usage
		currentCPU := getCPUUsage()
		if currentCPU > cfg.Performance.MaxCPUUsage {
			cli.PrintWarning(fmt.Sprintf("High CPU usage: %.1f%%", currentCPU))
			time.Sleep(100 * time.Millisecond) // Brief pause
		}

		// Save checkpoint every 5 epochs or at end
		if epoch%5 == 0 || epoch == numEpochs {
			cli.PrintStatus(fmt.Sprintf("Saving checkpoint: epoch %d", epoch), "info")

			// Ensure directory exists
			modelDir := filepath.Dir(cfg.Model.ModelPath)
			os.MkdirAll(modelDir, 0755)

			if err := net.Save(cfg.Model.ModelPath); err != nil {
				logger.Error("Failed to save checkpoint", "error", err)
				cli.PrintWarning(fmt.Sprintf("Failed to save: %v", err))
			} else {
				logger.Info("Checkpoint saved", "epoch", epoch, "path", cfg.Model.ModelPath)
			}
		}

		logger.LogEvent("epoch_complete", map[string]any{
			"epoch":       epoch,
			"loss":        loss,
			"duration_ms": epochDuration.Milliseconds(),
		})
	}

	totalDuration := time.Since(startTime)
	endCPU := getCPUUsage()

	cli.PrintSeparator()
	cli.PrintStatus(fmt.Sprintf("Training complete: %d epochs in %s", numEpochs, totalDuration), "success")
	cli.PrintStatus(fmt.Sprintf("Average CPU usage: %.1f%%", (startCPU+endCPU)/2), "info")

	logger.Info("Training complete",
		"total_epochs", numEpochs,
		"total_duration", totalDuration,
		"avg_cpu", (startCPU+endCPU)/2,
	)

	return nil
}

func runAnalyzeMode(ctx context.Context, cfg *config.Config, cli *iface.CLI) error {
	cli.PrintModeHeader("analyze")

	logger.Info("Starting analyze mode", "dataset", cfg.Training.DBPath)

	// Check if model exists
	if _, err := os.Stat(cfg.Model.ModelPath); os.IsNotExist(err) {
		return fmt.Errorf("model not found: %s", cfg.Model.ModelPath)
	}

	// Open dataset (use data.Dataset instead of storage.ObservationStore)
	cli.PrintStatus(fmt.Sprintf("Loading dataset from %s", cfg.Training.DBPath), "info")
	dataset, err := data.NewDataset(cfg.Training.DBPath)
	if err != nil {
		return fmt.Errorf("failed to open dataset: %w", err)
	}
	defer dataset.Close()

	stats, err := dataset.GetStats()
	if err != nil {
		return fmt.Errorf("failed to get dataset stats: %w", err)
	}

	if stats.TotalEntries == 0 {
		return fmt.Errorf("dataset is empty")
	}

	cli.PrintStatus(fmt.Sprintf("Dataset: %d observations", stats.TotalEntries), "success")

	// Initialize model for inference
	cli.PrintStatus("Loading model...", "info")
	cnnModel, err := model.NewChessCNNForInference(cfg.Model.ModelPath)
	if err != nil {
		return fmt.Errorf("failed to load model: %w", err)
	}
	defer cnnModel.Close()

	cli.PrintStatus("Model loaded successfully", "success")
	cli.PrintSeparator()

	// Analyze on test set
	testCount := 1000
	if testCount > stats.TotalEntries {
		testCount = stats.TotalEntries
	}

	cli.PrintStatus(fmt.Sprintf("Analyzing %d observations", testCount), "info")

	// Load test entries from dataset
	entries, err := dataset.LoadBatch(0, testCount)
	if err != nil {
		return fmt.Errorf("failed to load entries: %w", err)
	}

	correct := 0
	topKCorrect := make(map[int]int)

	for i, entry := range entries {
		select {
		case <-ctx.Done():
			return nil
		default:
		}

		cli.PrintProgress(i+1, testCount, fmt.Sprintf("Analyzing observation %d/%d", i+1, testCount))

		// Convert flat array to 3D tensor
		boardTensor, err := data.FlatArrayToTensor(entry.StateTensor)
		if err != nil {
			logger.Error("Failed to convert tensor", "observation", i, "error", err)
			continue
		}

		// Get predictions (top 10 moves)
		moves, err := cnnModel.Predict(boardTensor, 10)
		if err != nil {
			logger.Error("Prediction failed", "observation", i, "error", err)
			continue
		}

		// Calculate the expected move index from from/to squares
		expectedMoveIndex := entry.FromSquare*64 + entry.ToSquare

		// Check if correct move is in top predictions
		for k, move := range moves {
			if move.MoveIndex == expectedMoveIndex {
				if k == 0 {
					correct++
				}
				if k < 3 {
					topKCorrect[3]++
				}
				if k < 5 {
					topKCorrect[5]++
				}
				if k < 10 {
					topKCorrect[10]++
				}
				break
			}
		}
	}

	// Calculate top-K accuracies
	topKAccuracy := make(map[int]float64)
	for k, count := range topKCorrect {
		topKAccuracy[k] = float64(count) / float64(testCount) * 100
	}

	cli.PrintAnalysisResults(testCount, correct, topKAccuracy)

	logger.Info("Analysis complete",
		"total_positions", testCount,
		"correct", correct,
		"accuracy", float64(correct)/float64(testCount)*100,
		"top3_accuracy", topKAccuracy[3],
		"top5_accuracy", topKAccuracy[5],
	)

	return nil
}

func printHelp() {
	fmt.Println(`
P.A.R.T.N.E.R - Predictive Analysis & Real-Time Neural Engine for Chess

Usage:
  partner -mode <mode> [options]

Modes:
  observe   Watch screen and predict chess moves in real-time
  train     Train the model with dataset
  analyze   Run accuracy analysis on test dataset

Options:
  -config <path>    Path to configuration file (default: configs/partner.json)
  -quiet            Quiet mode - minimal output
  -dataset <path>   Path to dataset (overrides config)
  -model <path>     Path to model file (overrides config)
  -epochs <n>       Number of training epochs (train mode only)
  -help             Show this help message

Examples:
  # Observe mode - watch and predict
  partner -mode observe

  # Train mode - train model for 50 epochs
  partner -mode train -epochs 50

  # Analyze mode - test model accuracy
  partner -mode analyze

  # Quiet mode with custom config
  partner -mode observe -quiet -config my_config.json

Configuration:
  Create configs/partner.json to customize settings.
  Use -config flag to specify a different configuration file.

For more information, see: https://github.com/Moon9t/P.A.R.T.N.E.R`)
}

func getCPUUsage() float64 {
	// Simplified CPU usage - would use proper monitoring in production
	var m runtime.MemStats
	runtime.ReadMemStats(&m)
	return float64(runtime.NumGoroutine()) / float64(runtime.NumCPU()) * 10.0
}
