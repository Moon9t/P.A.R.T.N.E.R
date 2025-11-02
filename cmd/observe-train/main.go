package main

import (
	"flag"
	"fmt"
	"math/rand"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/thyrook/partner/internal/config"
	"github.com/thyrook/partner/internal/model"
	"github.com/thyrook/partner/internal/storage"
	"github.com/thyrook/partner/internal/training"
)

func main() {
	// Parse command line flags
	mode := flag.String("mode", "observe", "Mode: observe or train")
	configPath := flag.String("config", "config.json", "Path to configuration file")
	numSamples := flag.Int("samples", 100, "Number of samples to collect (observe mode)")
	epochs := flag.Int("epochs", 10, "Number of training epochs (train mode)")
	clearData := flag.Bool("clear", false, "Clear existing data before starting")
	flag.Parse()

	fmt.Println("═══════════════════════════════════════════════════════════")
	fmt.Println("  P.A.R.T.N.E.R - Data Collection & Training")
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

	fmt.Printf("✓ Observation store: %s\n", cfg.Training.DBPath)

	// Display current stats
	displayStats(store)
	fmt.Println()

	// Clear if requested
	if *clearData {
		fmt.Println("Clearing existing data...")
		if err := store.Clear(); err != nil {
			fmt.Fprintf(os.Stderr, "❌ Failed to clear store: %v\n", err)
			os.Exit(1)
		}
		fmt.Println("✓ Data cleared")
		fmt.Println()
	}

	// Run selected mode
	switch *mode {
	case "observe":
		runObserveMode(store, *numSamples)
	case "train":
		runTrainMode(store, cfg, *epochs)
	default:
		fmt.Fprintf(os.Stderr, "❌ Unknown mode: %s (use 'observe' or 'train')\n", *mode)
		os.Exit(1)
	}
}

// runObserveMode collects observation samples
func runObserveMode(store *storage.ObservationStore, numSamples int) {
	fmt.Printf("Mode: OBSERVE - Collecting %d samples\n", numSamples)
	fmt.Println("(Simulating chess observations)")
	fmt.Println("Press Ctrl+C to stop early")
	fmt.Println()

	// Setup signal handling
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, os.Interrupt, syscall.SIGTERM)

	collected := 0
	startTime := time.Now()

	ticker := time.NewTicker(100 * time.Millisecond)
	defer ticker.Stop()

	for collected < numSamples {
		select {
		case <-sigChan:
			fmt.Println("\n⚠️  Interrupted by user")
			goto done
		case <-ticker.C:
			// Generate simulated observation
			state, move := generateSimulatedObservation()

			if err := store.StoreSample(state, move); err != nil {
				fmt.Fprintf(os.Stderr, "❌ Failed to store: %v\n", err)
				continue
			}

			collected++

			if collected%10 == 0 || collected == numSamples {
				elapsed := time.Since(startTime)
				rate := float64(collected) / elapsed.Seconds()
				fmt.Printf("\rProgress: %d/%d (%.1f samples/sec)", collected, numSamples, rate)
			}
		}
	}

done:
	duration := time.Since(startTime)
	fmt.Printf("\n\n✓ Collected %d samples in %v\n", collected, duration.Round(time.Millisecond))
	fmt.Println()

	displayStats(store)

	fmt.Println()
	fmt.Println("═══════════════════════════════════════════════════════════")
	fmt.Println("  To train on this data, run:")
	fmt.Println("    ./bin/observe-train -mode=train -epochs=20")
	fmt.Println("═══════════════════════════════════════════════════════════")
}

// runTrainMode trains the model on collected data
func runTrainMode(store *storage.ObservationStore, cfg *config.Config, epochs int) {
	fmt.Println("Mode: TRAIN")
	fmt.Println()

	// Check if we have enough data
	count, err := store.CountSamples()
	if err != nil {
		fmt.Fprintf(os.Stderr, "❌ Failed to count samples: %v\n", err)
		os.Exit(1)
	}

	minSamples := uint64(cfg.Training.MinSamplesBeforeTraining)
	if count < minSamples {
		fmt.Fprintf(os.Stderr, "❌ Not enough samples: have %d, need %d\n", count, minSamples)
		fmt.Println("\nRun in observe mode first:")
		fmt.Printf("  ./bin/observe-train -mode=observe -samples=%d\n", minSamples)
		os.Exit(1)
	}

	fmt.Printf("✓ Found %d training samples\n", count)
	fmt.Println()

	// Initialize model
	fmt.Println("Initializing neural network...")
	net, err := model.NewChessNet(cfg.Model.InputSize, cfg.Model.HiddenSize, cfg.Model.OutputSize)
	if err != nil {
		fmt.Fprintf(os.Stderr, "❌ Failed to create model: %v\n", err)
		os.Exit(1)
	}
	defer net.Close()

	// Load existing weights if available
	if model.ModelExists(cfg.Model.ModelPath) {
		fmt.Printf("Loading existing model from %s...\n", cfg.Model.ModelPath)
		if err := net.Load(cfg.Model.ModelPath); err != nil {
			fmt.Printf("⚠️  Failed to load model, starting fresh: %v\n", err)
		} else {
			fmt.Println("✓ Model loaded")
		}
	} else {
		fmt.Println("Starting with fresh model")
	}

	fmt.Println()
	fmt.Printf("Training for %d epochs...\n", epochs)
	fmt.Println()

	// Configure training
	trainConfig := &training.TrainingConfig{
		Epochs:       epochs,
		BatchSize:    cfg.Model.BatchSize,
		LearningRate: cfg.Model.LearningRate,
		Verbose:      true,
	}

	// Train
	startTime := time.Now()

	if err := training.TrainFromStorage(net, store, trainConfig); err != nil {
		fmt.Fprintf(os.Stderr, "\n❌ Training failed: %v\n", err)
		os.Exit(1)
	}

	duration := time.Since(startTime)

	fmt.Println()
	fmt.Printf("✓ Training complete in %v\n", duration.Round(time.Millisecond))
	fmt.Printf("  Average: %.1f seconds/epoch\n", duration.Seconds()/float64(epochs))
	fmt.Println()

	// Save model
	fmt.Printf("Saving model to %s...\n", cfg.Model.ModelPath)
	if err := net.Save(cfg.Model.ModelPath); err != nil {
		fmt.Fprintf(os.Stderr, "❌ Failed to save model: %v\n", err)
		os.Exit(1)
	}

	// Get model size
	info, err := os.Stat(cfg.Model.ModelPath)
	if err == nil {
		fmt.Printf("✓ Model saved (%.2f KB)\n", float64(info.Size())/1024)
	} else {
		fmt.Println("✓ Model saved")
	}

	fmt.Println()
	fmt.Println("═══════════════════════════════════════════════════════════")
	fmt.Println("  Training Complete!")
	fmt.Println("  Model is ready for inference")
	fmt.Println("═══════════════════════════════════════════════════════════")
}

// generateSimulatedObservation generates a simulated chess observation
func generateSimulatedObservation() ([]float64, int) {
	state := make([]float64, 64)

	// Simulate a semi-realistic board state
	numPieces := 16 + rand.Intn(16)
	for i := 0; i < numPieces; i++ {
		square := rand.Intn(64)

		// Piece values: 0.1-0.6 white, 0.7-1.0 black
		if rand.Float64() < 0.5 {
			state[square] = 0.1 + rand.Float64()*0.5 // White
		} else {
			state[square] = 0.7 + rand.Float64()*0.3 // Black
		}
	}

	// Random move (from*64 + to)
	fromSquare := rand.Intn(64)
	toSquare := rand.Intn(64)
	move := fromSquare*64 + toSquare

	return state, move
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

func init() {
	rand.Seed(time.Now().UnixNano())
}
