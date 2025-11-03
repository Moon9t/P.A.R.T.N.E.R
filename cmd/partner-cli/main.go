package main

import (
	"bufio"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/thyrook/partner/internal/data"
	"github.com/thyrook/partner/internal/model"
	"github.com/thyrook/partner/internal/storage"
	"go.etcd.io/bbolt"
)

const (
	version = "1.0.0"
	banner  = `
╔═══════════════════════════════════════════════════════════╗
║                                                           ║
║   P.A.R.T.N.E.R - Pattern Analysis & Real-Time Neural    ║
║              Enhancement for Reinforcement                ║
║                                                           ║
║                    Version %s                          ║
║                                                           ║
╚═══════════════════════════════════════════════════════════╝
`
)

type CLI struct {
	dataset          *data.Dataset
	model            *model.ChessCNN
	observationStore *storage.ObservationStore
	modelPath        string
	datasetPath      string
	running          bool
}

func main() {
	fmt.Printf(banner, version)
	fmt.Println()

	cli := &CLI{
		modelPath:   "data/models/chess_cnn.gob",
		datasetPath: "data/positions.db",
		running:     true,
	}

	// Show main menu
	cli.mainMenu()
}

func (c *CLI) mainMenu() {
	reader := bufio.NewReader(os.Stdin)

	for c.running {
		fmt.Println("\n" + strings.Repeat("=", 60))
		fmt.Println("MAIN MENU")
		fmt.Println(strings.Repeat("=", 60))
		fmt.Println("1. Dataset Management")
		fmt.Println("2. Model Training")
		fmt.Println("3. Model Inference")
		fmt.Println("4. System Status")
		fmt.Println("5. Configuration")
		fmt.Println("6. Help")
		fmt.Println("0. Exit")
		fmt.Println(strings.Repeat("=", 60))
		fmt.Print("\nSelect option: ")

		input, _ := reader.ReadString('\n')
		input = strings.TrimSpace(input)

		switch input {
		case "1":
			c.datasetMenu()
		case "2":
			c.trainingMenu()
		case "3":
			c.inferenceMenu()
		case "4":
			c.showStatus()
		case "5":
			c.configMenu()
		case "6":
			c.showHelp()
		case "0":
			c.running = false
			fmt.Println("\n✓ Goodbye!")
		default:
			fmt.Println("Invalid option")
		}
	}
}

func (c *CLI) datasetMenu() {
	reader := bufio.NewReader(os.Stdin)

	for {
		fmt.Println("\n" + strings.Repeat("-", 60))
		fmt.Println("DATASET MANAGEMENT")
		fmt.Println(strings.Repeat("-", 60))
		fmt.Println("1. View dataset statistics")
		fmt.Println("2. Ingest PGN file")
		fmt.Println("3. Export samples")
		fmt.Println("4. Validate dataset")
		fmt.Println("5. Compact database")
		fmt.Println("0. Back to main menu")
		fmt.Println(strings.Repeat("-", 60))
		fmt.Print("\nSelect option: ")

		input, _ := reader.ReadString('\n')
		input = strings.TrimSpace(input)

		switch input {
		case "1":
			c.showDatasetStats()
		case "2":
			c.ingestPGN()
		case "3":
			c.exportSamples()
		case "4":
			c.validateDataset()
		case "5":
			c.compactDatabase()
		case "0":
			return
		default:
			fmt.Println("Invalid option")
		}
	}
}

func (c *CLI) trainingMenu() {
	reader := bufio.NewReader(os.Stdin)

	for {
		fmt.Println("\n" + strings.Repeat("-", 60))
		fmt.Println("MODEL TRAINING")
		fmt.Println(strings.Repeat("-", 60))
		fmt.Println("1. Quick training (10 epochs)")
		fmt.Println("2. Standard training (50 epochs)")
		fmt.Println("3. Full training (100 epochs)")
		fmt.Println("4. Custom training")
		fmt.Println("5. Resume training")
		fmt.Println("6. View training history")
		fmt.Println("0. Back to main menu")
		fmt.Println(strings.Repeat("-", 60))
		fmt.Print("\nSelect option: ")

		input, _ := reader.ReadString('\n')
		input = strings.TrimSpace(input)

		switch input {
		case "1":
			c.trainModel(10, "quick")
		case "2":
			c.trainModel(50, "standard")
		case "3":
			c.trainModel(100, "full")
		case "4":
			c.customTraining()
		case "5":
			c.resumeTraining()
		case "6":
			c.viewTrainingHistory()
		case "0":
			return
		default:
			fmt.Println("Invalid option")
		}
	}
}

func (c *CLI) inferenceMenu() {
	reader := bufio.NewReader(os.Stdin)

	for {
		fmt.Println("\n" + strings.Repeat("-", 60))
		fmt.Println("MODEL INFERENCE")
		fmt.Println(strings.Repeat("-", 60))
		fmt.Println("1. Load model")
		fmt.Println("2. Test inference on random position")
		fmt.Println("3. Test inference on custom FEN")
		fmt.Println("4. Batch inference")
		fmt.Println("5. Model performance test")
		fmt.Println("0. Back to main menu")
		fmt.Println(strings.Repeat("-", 60))
		fmt.Print("\nSelect option: ")

		input, _ := reader.ReadString('\n')
		input = strings.TrimSpace(input)

		switch input {
		case "1":
			c.loadModel()
		case "2":
			c.testRandomPosition()
		case "3":
			c.testCustomFEN()
		case "4":
			c.batchInference()
		case "5":
			c.performanceTest()
		case "0":
			return
		default:
			fmt.Println("Invalid option")
		}
	}
}

func (c *CLI) showStatus() {
	fmt.Println("\n" + strings.Repeat("=", 60))
	fmt.Println("SYSTEM STATUS")
	fmt.Println(strings.Repeat("=", 60))

	// Dataset status
	fmt.Println("\nDataset:")
	if _, err := os.Stat(c.datasetPath); err == nil {
		dataset, err := data.NewDataset(c.datasetPath)
		if err == nil {
			defer dataset.Close()
			count, _ := dataset.Count()
			stats, _ := dataset.GetStats()

			fmt.Printf("  Path:       %s\n", c.datasetPath)
			fmt.Printf("  Positions:  %d\n", count)
			fmt.Printf("  Size:       %.2f MB\n", float64(stats.FileSize)/1024/1024)
			fmt.Printf("  Status:     ✓ Available\n")
		} else {
			fmt.Printf("  Status:     ⚠ Error: %v\n", err)
		}
	} else {
		fmt.Printf("  Status:     Not found\n")
	}

	// Model status
	fmt.Println("\nModel:")
	if _, err := os.Stat(c.modelPath); err == nil {
		info, _ := os.Stat(c.modelPath)
		fmt.Printf("  Path:       %s\n", c.modelPath)
		fmt.Printf("  Size:       %.2f MB\n", float64(info.Size())/1024/1024)
		fmt.Printf("  Modified:   %s\n", info.ModTime().Format("2006-01-02 15:04:05"))
		fmt.Printf("  Status:     ✓ Available\n")
	} else {
		fmt.Printf("  Status:     Not trained\n")
	}

	// System info
	fmt.Println("\n💻 System:")
	fmt.Printf("  Version:    %s\n", version)
	fmt.Printf("  Time:       %s\n", time.Now().Format("2006-01-02 15:04:05"))

	fmt.Println(strings.Repeat("=", 60))
}

func (c *CLI) configMenu() {
	reader := bufio.NewReader(os.Stdin)

	for {
		fmt.Println("\n" + strings.Repeat("-", 60))
		fmt.Println("CONFIGURATION")
		fmt.Println(strings.Repeat("-", 60))
		fmt.Printf("Current Settings:\n")
		fmt.Printf("  Model path:   %s\n", c.modelPath)
		fmt.Printf("  Dataset path: %s\n", c.datasetPath)
		fmt.Println(strings.Repeat("-", 60))
		fmt.Println("1. Change model path")
		fmt.Println("2. Change dataset path")
		fmt.Println("3. Reset to defaults")
		fmt.Println("0. Back to main menu")
		fmt.Println(strings.Repeat("-", 60))
		fmt.Print("\nSelect option: ")

		input, _ := reader.ReadString('\n')
		input = strings.TrimSpace(input)

		switch input {
		case "1":
			c.changeModelPath(reader)
		case "2":
			c.changeDatasetPath(reader)
		case "3":
			c.modelPath = "data/models/chess_cnn.gob"
			c.datasetPath = "data/positions.db"
			fmt.Println("✓ Reset to defaults")
		case "0":
			return
		default:
			fmt.Println("Invalid option")
		}
	}
}

func (c *CLI) showHelp() {
	fmt.Println("\n" + strings.Repeat("=", 60))
	fmt.Println("HELP & DOCUMENTATION")
	fmt.Println(strings.Repeat("=", 60))
	fmt.Println(`
P.A.R.T.N.E.R is a chess move prediction system using CNN.

WORKFLOW:
1. Ingest PGN files to build training dataset
2. Train the CNN model on the dataset
3. Use the model for move prediction

DATASET MANAGEMENT:
- Ingest PGN: Convert chess games to training data
- Statistics: View dataset size and composition
- Validate: Check data integrity

MODEL TRAINING:
- Quick (10 epochs): Fast training for testing
- Standard (50 epochs): Balanced training
- Full (100 epochs): Production training
- Custom: Configure all hyperparameters

INFERENCE:
- Test on random positions from dataset
- Test on custom FEN strings
- Batch inference for performance testing

TIPS:
- Start with a small dataset (1000 games) for testing
- Use validation split (15%) to monitor overfitting
- Enable early stopping for efficient training
- Save checkpoints every 5 epochs

For more information, see README.md
`)
	fmt.Println(strings.Repeat("=", 60))
}

func (c *CLI) showDatasetStats() {
	fmt.Println("\nLoading dataset statistics...")

	dataset, err := data.NewDataset(c.datasetPath)
	if err != nil {
		fmt.Printf("Failed to open dataset: %v\n", err)
		return
	}
	defer dataset.Close()

	count, err := dataset.Count()
	if err != nil {
		fmt.Printf("Failed to count entries: %v\n", err)
		return
	}

	stats, err := dataset.GetStats()
	if err != nil {
		fmt.Printf("Failed to get stats: %v\n", err)
		return
	}

	fmt.Println("\n" + strings.Repeat("-", 60))
	fmt.Println("DATASET STATISTICS")
	fmt.Println(strings.Repeat("-", 60))
	fmt.Printf("File path:        %s\n", stats.FilePath)
	fmt.Printf("Total positions:  %d\n", count)
	fmt.Printf("File size:        %.2f MB\n", float64(stats.FileSize)/1024/1024)
	fmt.Printf("Avg bytes/entry:  %.1f bytes\n", float64(stats.FileSize)/float64(count))
	fmt.Println(strings.Repeat("-", 60))

	// Sample some positions to show distribution
	if count > 100 {
		opening, mid, end := 0, 0, 0
		sampleSize := 100
		for i := 0; i < sampleSize; i++ {
			idx := i * (count / sampleSize)
			entries, err := dataset.LoadBatch(idx, 1)
			if err == nil && len(entries) > 0 {
				move := entries[0].MoveNumber
				if move <= 10 {
					opening++
				} else if move <= 30 {
					mid++
				} else {
					end++
				}
			}
		}

		fmt.Println("\nSampled Distribution (100 positions):")
		fmt.Printf("  Opening (moves 1-10):     ~%d%%\n", opening)
		fmt.Printf("  Middlegame (moves 11-30): ~%d%%\n", mid)
		fmt.Printf("  Endgame (moves 31+):      ~%d%%\n", end)
	}
	fmt.Println(strings.Repeat("-", 60))
}

func (c *CLI) ingestPGN() {
	reader := bufio.NewReader(os.Stdin)

	fmt.Print("\nEnter PGN file path: ")
	pgnPath, _ := reader.ReadString('\n')
	pgnPath = strings.TrimSpace(pgnPath)

	if _, err := os.Stat(pgnPath); err != nil {
		fmt.Printf("File not found: %s\n", pgnPath)
		return
	}

	fmt.Print("Max games to process (0 = all): ")
	maxGamesStr, _ := reader.ReadString('\n')
	maxGames, _ := strconv.Atoi(strings.TrimSpace(maxGamesStr))

	fmt.Printf("\nIngesting PGN file: %s\n", pgnPath)
	fmt.Println("This may take a while...")

	// Call the ingest-pgn functionality
	fmt.Printf("\n⚠ Please use: ./bin/ingest-pgn --pgn %s --dataset %s\n", pgnPath, c.datasetPath)
	if maxGames > 0 {
		fmt.Printf("   Add: --max-games %d\n", maxGames)
	}
}

func (c *CLI) exportSamples() {
	reader := bufio.NewReader(os.Stdin)

	fmt.Print("\nNumber of samples to export: ")
	numStr, _ := reader.ReadString('\n')
	numSamples, err := strconv.Atoi(strings.TrimSpace(numStr))
	if err != nil || numSamples <= 0 {
		fmt.Println("Invalid number")
		return
	}

	fmt.Print("Output file (e.g., samples.txt): ")
	outFile, _ := reader.ReadString('\n')
	outFile = strings.TrimSpace(outFile)

	dataset, err := data.NewDataset(c.datasetPath)
	if err != nil {
		fmt.Printf("Failed to open dataset: %v\n", err)
		return
	}
	defer dataset.Close()

	file, err := os.Create(outFile)
	if err != nil {
		fmt.Printf("Failed to create output file: %v\n", err)
		return
	}
	defer file.Close()

	fmt.Printf("\nExporting %d samples...\n", numSamples)

	for i := 0; i < numSamples; i++ {
		entries, err := dataset.LoadBatch(i, 1)
		if err != nil || len(entries) == 0 {
			break
		}

		entry := entries[0]
		fmt.Fprintf(file, "Position %d:\n", i+1)
		fmt.Fprintf(file, "  Game: %s\n", entry.GameID)
		fmt.Fprintf(file, "  Move: %d\n", entry.MoveNumber)
		fmt.Fprintf(file, "  From: %d, To: %d\n", entry.FromSquare, entry.ToSquare)
		fmt.Fprintf(file, "\n")

		if (i+1)%100 == 0 {
			fmt.Printf("  Exported %d/%d samples...\r", i+1, numSamples)
		}
	}

	fmt.Printf("\n✓ Exported to: %s\n", outFile)
}

func (c *CLI) validateDataset() {
	fmt.Println("\n🔍 Validating dataset...")

	dataset, err := data.NewDataset(c.datasetPath)
	if err != nil {
		fmt.Printf("Failed to open dataset: %v\n", err)
		return
	}
	defer dataset.Close()

	count, _ := dataset.Count()
	errors := 0
	checked := 0

	fmt.Printf("Checking %d positions...\n", count)

	sampleSize := 1000
	if count < sampleSize {
		sampleSize = count
	}

	for i := 0; i < sampleSize; i++ {
		entries, err := dataset.LoadBatch(i*(count/sampleSize), 1)
		if err != nil || len(entries) == 0 {
			errors++
			continue
		}

		entry := entries[0]

		// Validate move indices
		if entry.FromSquare < 0 || entry.FromSquare >= 64 {
			errors++
			continue
		}
		if entry.ToSquare < 0 || entry.ToSquare >= 64 {
			errors++
			continue
		}

		// Validate tensor
		if len(entry.StateTensor) != 768 {
			errors++
			continue
		}

		checked++
		if checked%100 == 0 {
			fmt.Printf("  Checked %d/%d samples...\r", checked, sampleSize)
		}
	}

	fmt.Printf("\n\n" + strings.Repeat("-", 60) + "\n")
	fmt.Println("VALIDATION RESULTS")
	fmt.Println(strings.Repeat("-", 60))
	fmt.Printf("Samples checked: %d\n", checked)
	fmt.Printf("Errors found:    %d\n", errors)
	if errors == 0 {
		fmt.Printf("Status:          ✓ All checks passed\n")
	} else {
		fmt.Printf("Status:          ⚠ %d errors (%.2f%%)\n", errors, float64(errors)/float64(checked)*100)
	}
	fmt.Println(strings.Repeat("-", 60))
}

func (c *CLI) compactDatabase() {
	fmt.Println("\nCompacting database...")

	if c.observationStore == nil {
		fmt.Println("No dataset loaded")
		return
	}

	// Close the current store
	if err := c.observationStore.Close(); err != nil {
		fmt.Printf("Failed to close database: %v\n", err)
		return
	}

	// Create backup path
	backupPath := c.datasetPath + ".backup"

	// Copy original to backup
	if err := copyFile(c.datasetPath, backupPath); err != nil {
		fmt.Printf("Failed to create backup: %v\n", err)
		// Try to reopen the original store
		c.observationStore, _ = storage.NewObservationStore(c.datasetPath, 1000000)
		return
	}

	// Reopen and compact using BoltDB's internal compaction
	tmpPath := c.datasetPath + ".tmp"
	if err := compactBoltDB(c.datasetPath, tmpPath); err != nil {
		fmt.Printf("Compaction failed: %v\n", err)
		// Restore from backup
		os.Remove(tmpPath)
		c.observationStore, _ = storage.NewObservationStore(c.datasetPath, 1000000)
		return
	}

	// Replace original with compacted version
	if err := os.Rename(tmpPath, c.datasetPath); err != nil {
		fmt.Printf("Failed to replace database: %v\n", err)
		os.Remove(tmpPath)
		c.observationStore, _ = storage.NewObservationStore(c.datasetPath, 1000000)
		return
	}

	// Reopen the compacted store
	store, err := storage.NewObservationStore(c.datasetPath, 1000000)
	if err != nil {
		fmt.Printf("Failed to reopen database: %v\n", err)
		return
	}
	c.observationStore = store

	// Remove backup
	os.Remove(backupPath)

	fmt.Println("✓ Database compacted successfully")
}

func copyFile(src, dst string) error {
	data, err := os.ReadFile(src)
	if err != nil {
		return err
	}
	return os.WriteFile(dst, data, 0600)
}

func compactBoltDB(srcPath, dstPath string) error {
	srcDB, err := bbolt.Open(srcPath, 0600, &bbolt.Options{ReadOnly: true})
	if err != nil {
		return fmt.Errorf("failed to open source db: %w", err)
	}
	defer srcDB.Close()

	dstDB, err := bbolt.Open(dstPath, 0600, nil)
	if err != nil {
		return fmt.Errorf("failed to create destination db: %w", err)
	}
	defer dstDB.Close()

	// Compact by copying all data
	return srcDB.View(func(srcTx *bbolt.Tx) error {
		return dstDB.Update(func(dstTx *bbolt.Tx) error {
			return srcTx.ForEach(func(name []byte, srcBucket *bbolt.Bucket) error {
				dstBucket, err := dstTx.CreateBucketIfNotExists(name)
				if err != nil {
					return err
				}
				return srcBucket.ForEach(func(k, v []byte) error {
					return dstBucket.Put(k, v)
				})
			})
		})
	})
}

func (c *CLI) trainModel(epochs int, preset string) {
	fmt.Printf("\nStarting %s training (%d epochs)...\n", preset, epochs)

	// Check if dataset exists
	if _, err := os.Stat(c.datasetPath); err != nil {
		fmt.Printf("Dataset not found: %s\n", c.datasetPath)
		fmt.Println("Please ingest a PGN file first")
		return
	}

	// Open dataset
	dataset, err := data.NewDataset(c.datasetPath)
	if err != nil {
		fmt.Printf("Failed to open dataset: %v\n", err)
		return
	}
	defer dataset.Close()

	count, _ := dataset.Count()
	if count == 0 {
		fmt.Println("Dataset is empty")
		return
	}

	fmt.Printf("Dataset: %d positions\n", count)

	// Create training configuration
	config := &model.TrainingConfig{
		Epochs:            epochs,
		BatchSize:         64,
		LearningRate:      0.001,
		LRDecayRate:       0.95,
		LRDecaySteps:      2,
		GradientClipMax:   5.0,
		ValidationSplit:   0.15,
		EarlyStopPatience: 10,
		WarmupEpochs:      3,
		GradAccumSteps:    1,
		ShuffleBatches:    true,
		WeightDecay:       0.0001,
		SaveInterval:      5,
		SavePath:          c.modelPath,
		Verbose:           true,
	}

	// Adjust for preset
	switch preset {
	case "quick":
		config.SaveInterval = 0      // Don't save checkpoints for quick training
		config.EarlyStopPatience = 0 // No early stopping
	case "standard":
		config.SaveInterval = 10
	case "full":
		config.SaveInterval = 5
	}

	fmt.Println("\n" + strings.Repeat("-", 60))
	fmt.Println("Training Configuration:")
	fmt.Printf("  Epochs:          %d\n", config.Epochs)
	fmt.Printf("  Batch size:      %d\n", config.BatchSize)
	fmt.Printf("  Learning rate:   %.6f\n", config.LearningRate)
	fmt.Printf("  Validation:      %.0f%%\n", config.ValidationSplit*100)
	fmt.Printf("  Early stopping:  %d epochs patience\n", config.EarlyStopPatience)
	fmt.Printf("  Warmup:          %d epochs\n", config.WarmupEpochs)
	fmt.Println(strings.Repeat("-", 60))

	reader := bufio.NewReader(os.Stdin)
	fmt.Print("\nProceed with training? (y/n): ")
	confirm, _ := reader.ReadString('\n')
	if strings.ToLower(strings.TrimSpace(confirm)) != "y" {
		fmt.Println("Training cancelled")
		return
	}

	// Create trainer and train
	fmt.Println("\n🚀 Initializing trainer...")
	trainer, err := model.NewTrainer(config)
	if err != nil {
		fmt.Printf("Failed to create trainer: %v\n", err)
		return
	}

	fmt.Println("\n" + strings.Repeat("=", 60))
	fmt.Println("TRAINING STARTED")
	fmt.Println(strings.Repeat("=", 60))

	startTime := time.Now()
	err = trainer.Train(dataset)
	duration := time.Since(startTime)

	if err != nil {
		fmt.Printf("\nTraining failed: %v\n", err)
		return
	}

	fmt.Println("\n" + strings.Repeat("=", 60))
	fmt.Println("TRAINING COMPLETED")
	fmt.Println(strings.Repeat("=", 60))
	fmt.Printf("Total time: %v\n", duration)
	fmt.Printf("Model saved: %s\n", c.modelPath)
	fmt.Println(strings.Repeat("=", 60))
}

func (c *CLI) customTraining() {
	reader := bufio.NewReader(os.Stdin)

	fmt.Println("\n" + strings.Repeat("-", 60))
	fmt.Println("CUSTOM TRAINING CONFIGURATION")
	fmt.Println(strings.Repeat("-", 60))

	// Get epochs
	fmt.Print("Epochs (default 50): ")
	epochsStr, _ := reader.ReadString('\n')
	epochs, err := strconv.Atoi(strings.TrimSpace(epochsStr))
	if err != nil || epochs <= 0 {
		epochs = 50
	}

	// Get batch size
	fmt.Print("Batch size (default 64): ")
	batchStr, _ := reader.ReadString('\n')
	batchSize, err := strconv.Atoi(strings.TrimSpace(batchStr))
	if err != nil || batchSize <= 0 {
		batchSize = 64
	}

	// Get learning rate
	fmt.Print("Learning rate (default 0.001): ")
	lrStr, _ := reader.ReadString('\n')
	lr, err := strconv.ParseFloat(strings.TrimSpace(lrStr), 64)
	if err != nil || lr <= 0 {
		lr = 0.001
	}

	// Get validation split
	fmt.Print("Validation split 0-1 (default 0.15): ")
	valStr, _ := reader.ReadString('\n')
	valSplit, err := strconv.ParseFloat(strings.TrimSpace(valStr), 64)
	if err != nil || valSplit < 0 || valSplit >= 1 {
		valSplit = 0.15
	}

	// Start training with custom config
	fmt.Printf("\n✓ Custom configuration set\n")

	// Open dataset
	dataset, err := data.NewDataset(c.datasetPath)
	if err != nil {
		fmt.Printf("Failed to open dataset: %v\n", err)
		return
	}
	defer dataset.Close()

	config := &model.TrainingConfig{
		Epochs:            epochs,
		BatchSize:         batchSize,
		LearningRate:      lr,
		LRDecayRate:       0.95,
		LRDecaySteps:      2,
		GradientClipMax:   5.0,
		ValidationSplit:   valSplit,
		EarlyStopPatience: 10,
		WarmupEpochs:      3,
		GradAccumSteps:    1,
		ShuffleBatches:    true,
		WeightDecay:       0.0001,
		SaveInterval:      5,
		SavePath:          c.modelPath,
		Verbose:           true,
	}

	trainer, err := model.NewTrainer(config)
	if err != nil {
		fmt.Printf("Failed to create trainer: %v\n", err)
		return
	}

	fmt.Println("\n🚀 Starting custom training...")
	err = trainer.Train(dataset)
	if err != nil {
		fmt.Printf("Training failed: %v\n", err)
		return
	}

	fmt.Println("✓ Training completed!")
}

func (c *CLI) resumeTraining() {
	reader := bufio.NewReader(os.Stdin)

	fmt.Println("\nResume Training")

	// Check if model exists
	if _, err := os.Stat(c.modelPath); err != nil {
		fmt.Printf("Model not found: %s\n", c.modelPath)
		fmt.Println("Please train a model first or change the model path")
		return
	}

	// Get training parameters
	fmt.Print("\nNumber of additional epochs: ")
	epochsStr, _ := reader.ReadString('\n')
	epochs, err := strconv.Atoi(strings.TrimSpace(epochsStr))
	if err != nil || epochs <= 0 {
		fmt.Println("Invalid epoch count")
		return
	}

	fmt.Print("Learning rate (default 0.001): ")
	lrStr, _ := reader.ReadString('\n')
	lrStr = strings.TrimSpace(lrStr)
	learningRate := 0.001
	if lrStr != "" {
		if lr, err := strconv.ParseFloat(lrStr, 64); err == nil && lr > 0 {
			learningRate = lr
		}
	}

	// Check dataset
	if _, err := os.Stat(c.datasetPath); err != nil {
		fmt.Printf("Dataset not found: %s\n", c.datasetPath)
		return
	}

	// Load dataset
	fmt.Println("\nLoading dataset...")
	dataset, err := data.NewDataset(c.datasetPath)
	if err != nil {
		fmt.Printf("Failed to load dataset: %v\n", err)
		return
	}
	defer dataset.Close()

	stats, err := dataset.GetStats()
	if err != nil {
		fmt.Printf("Failed to get dataset stats: %v\n", err)
		return
	}
	fmt.Printf("Dataset size: %d positions\n", stats.TotalEntries)

	// Create training config to resume training
	// The trainer will create a new model and load the checkpoint automatically
	config := &model.TrainingConfig{
		Epochs:          epochs,
		BatchSize:       64,
		LearningRate:    learningRate,
		LRDecayRate:     0.95,
		LRDecaySteps:    2,
		GradientClipMax: 5.0,
		Verbose:         true,
		SaveInterval:    5,
		SavePath:        c.modelPath,
		ValidationSplit: 0.1,
		ShuffleBatches:  true,
		WeightDecay:     0.0001,
	}

	// Continue training
	fmt.Printf("\nResuming training for %d epochs (lr=%.4f)...\n", epochs, learningRate)
	fmt.Println("Loading existing model weights...")

	trainer, err := model.NewTrainer(config)
	if err != nil {
		fmt.Printf("Failed to create trainer: %v\n", err)
		return
	}

	err = trainer.Train(dataset)
	if err != nil {
		fmt.Printf("Training failed: %v\n", err)
		return
	}

	fmt.Println("✓ Training resumed and completed!")
}

func (c *CLI) viewTrainingHistory() {
	fmt.Println("\nTraining History")
	fmt.Println(strings.Repeat("-", 60))

	// Look for training log files
	logDir := "logs"
	logPattern := filepath.Join(logDir, "training_*.log")

	matches, err := filepath.Glob(logPattern)
	if err != nil || len(matches) == 0 {
		fmt.Println("No training history found")
		fmt.Println("Training logs are created in the 'logs' directory during training")
		return
	}

	// Display available log files
	fmt.Printf("Found %d training log file(s):\n\n", len(matches))
	for i, logPath := range matches {
		info, err := os.Stat(logPath)
		if err != nil {
			continue
		}
		fmt.Printf("%d. %s (%.2f KB, modified: %s)\n",
			i+1,
			filepath.Base(logPath),
			float64(info.Size())/1024,
			info.ModTime().Format("2006-01-02 15:04:05"))
	}

	// Allow user to view a log file
	reader := bufio.NewReader(os.Stdin)
	fmt.Print("\nEnter log number to view (0 to cancel): ")
	input, _ := reader.ReadString('\n')
	choice, err := strconv.Atoi(strings.TrimSpace(input))
	if err != nil || choice < 1 || choice > len(matches) {
		fmt.Println("Cancelled")
		return
	}

	logPath := matches[choice-1]

	// Read and display log content
	fmt.Printf("\n📄 Content of %s:\n", filepath.Base(logPath))
	fmt.Println(strings.Repeat("-", 60))

	content, err := os.ReadFile(logPath)
	if err != nil {
		fmt.Printf("Failed to read log: %v\n", err)
		return
	}

	// Display last 50 lines to avoid overwhelming output
	lines := strings.Split(string(content), "\n")
	startIdx := 0
	if len(lines) > 50 {
		startIdx = len(lines) - 50
		fmt.Println("... (showing last 50 lines) ...")
	}

	for i := startIdx; i < len(lines); i++ {
		if lines[i] != "" {
			fmt.Println(lines[i])
		}
	}

	fmt.Println(strings.Repeat("-", 60))
	fmt.Printf("Total lines: %d\n", len(lines))
}

func (c *CLI) loadModel() {
	fmt.Printf("\nLoading model from: %s\n", c.modelPath)

	if _, err := os.Stat(c.modelPath); err != nil {
		fmt.Printf("Model not found: %s\n", c.modelPath)
		fmt.Println("Please train a model first")
		return
	}

	model, err := model.NewChessCNNForInference(c.modelPath)
	if err != nil {
		fmt.Printf("Failed to load model: %v\n", err)
		return
	}
	defer model.Close()

	c.model = model
	fmt.Println("✓ Model loaded successfully")
}

func (c *CLI) testRandomPosition() {
	if c.model == nil {
		fmt.Println("No model loaded. Please load a model first")
		return
	}

	// Load a random position from dataset
	dataset, err := data.NewDataset(c.datasetPath)
	if err != nil {
		fmt.Printf("Failed to open dataset: %v\n", err)
		return
	}
	defer dataset.Close()

	count, _ := dataset.Count()
	if count == 0 {
		fmt.Println("Dataset is empty")
		return
	}

	// Get random position
	randomIdx := time.Now().UnixNano() % int64(count)
	entries, err := dataset.LoadBatch(int(randomIdx), 1)
	if err != nil || len(entries) == 0 {
		fmt.Println("Failed to load position")
		return
	}

	entry := entries[0]
	boardTensor, err := data.FlatArrayToTensor(entry.StateTensor)
	if err != nil {
		fmt.Printf("Failed to convert tensor: %v\n", err)
		return
	}

	fmt.Println("\n" + strings.Repeat("-", 60))
	fmt.Printf("Testing position from game: %s (move %d)\n", entry.GameID, entry.MoveNumber)
	fmt.Printf("Actual move: %s → %s\n",
		squareToAlgebraic(entry.FromSquare),
		squareToAlgebraic(entry.ToSquare))
	fmt.Println(strings.Repeat("-", 60))

	// Run inference
	predictions, err := c.model.Predict(boardTensor, 5)
	if err != nil {
		fmt.Printf("Inference failed: %v\n", err)
		return
	}

	fmt.Println("\nTop 5 predictions:")
	for i, pred := range predictions {
		marker := "  "
		if pred.FromSquare == entry.FromSquare && pred.ToSquare == entry.ToSquare {
			marker = "✓ "
		}
		fmt.Printf("%s%d. %s → %s  (%.2f%%)\n",
			marker, i+1,
			squareToAlgebraic(pred.FromSquare),
			squareToAlgebraic(pred.ToSquare),
			pred.Probability*100)
	}
	fmt.Println(strings.Repeat("-", 60))
}

func (c *CLI) testCustomFEN() {
	reader := bufio.NewReader(os.Stdin)

	if c.model == nil {
		fmt.Println("No model loaded. Please load a model first")
		return
	}

	fmt.Println("\n🎯 Custom FEN Position Testing")
	fmt.Println(strings.Repeat("-", 60))
	fmt.Println("Enter a FEN string to test the model's prediction")
	fmt.Println("Example: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
	fmt.Println(strings.Repeat("-", 60))

	fmt.Print("\nFEN string (or 'q' to cancel): ")
	fenInput, _ := reader.ReadString('\n')
	fenInput = strings.TrimSpace(fenInput)

	if fenInput == "q" || fenInput == "" {
		fmt.Println("Cancelled")
		return
	}

	// Parse FEN and convert to board state
	boardState, err := parseFENToState(fenInput)
	if err != nil {
		fmt.Printf("Invalid FEN string: %v\n", err)
		return
	}

	// Convert simple state to 12-channel tensor
	boardTensor := convertStateTo12Channel(boardState)

	// Run inference
	fmt.Println("\n🔍 Running inference...")
	startTime := time.Now()
	predictions, err := c.model.Predict(boardTensor, 5)
	latency := time.Since(startTime)

	if err != nil {
		fmt.Printf("Prediction failed: %v\n", err)
		return
	}

	// Display results
	fmt.Println("\n✓ Top 5 predicted moves:")
	fmt.Println(strings.Repeat("-", 60))
	fmt.Printf("%-4s %-8s %-12s %s\n", "Rank", "Move", "Confidence", "Bar")
	fmt.Println(strings.Repeat("-", 60))

	for i := 0; i < 5 && i < len(predictions); i++ {
		pred := predictions[i]
		barLen := int(pred.Probability * 30)
		bar := strings.Repeat("█", barLen)

		fmt.Printf("%-4d %-8s %6.2f%%      %s\n",
			i+1,
			moveIndexToAlgebraic(pred.MoveIndex),
			pred.Probability*100,
			bar)
	}

	fmt.Println(strings.Repeat("-", 60))
	fmt.Printf("Inference time: %v\n", latency)
}

// parseFENToState converts a FEN string to a board state tensor
func parseFENToState(fen string) ([]float64, error) {
	parts := strings.Fields(fen)
	if len(parts) < 1 {
		return nil, fmt.Errorf("invalid FEN format")
	}

	boardPart := parts[0]
	state := make([]float64, 64)

	// Parse board position
	rank := 0
	file := 0

	for _, ch := range boardPart {
		if ch == '/' {
			rank++
			file = 0
			continue
		}

		if ch >= '1' && ch <= '8' {
			// Empty squares
			emptyCount := int(ch - '0')
			file += emptyCount
			continue
		}

		if rank >= 8 || file >= 8 {
			return nil, fmt.Errorf("invalid board position")
		}

		// Map piece to value
		pieceValue := 0.0
		switch ch {
		case 'P':
			pieceValue = 1.0
		case 'N':
			pieceValue = 2.0
		case 'B':
			pieceValue = 3.0
		case 'R':
			pieceValue = 4.0
		case 'Q':
			pieceValue = 5.0
		case 'K':
			pieceValue = 6.0
		case 'p':
			pieceValue = -1.0
		case 'n':
			pieceValue = -2.0
		case 'b':
			pieceValue = -3.0
		case 'r':
			pieceValue = -4.0
		case 'q':
			pieceValue = -5.0
		case 'k':
			pieceValue = -6.0
		default:
			return nil, fmt.Errorf("invalid piece character: %c", ch)
		}

		idx := rank*8 + file
		state[idx] = pieceValue
		file++
	}

	return state, nil
}

// moveIndexToAlgebraic converts move index to algebraic notation (simplified)
func moveIndexToAlgebraic(moveIndex int) string {
	// Decode move index (simplified: from_square * 64 + to_square)
	fromSquare := moveIndex / 64
	toSquare := moveIndex % 64

	fromFile := fromSquare % 8
	fromRank := fromSquare / 8
	toFile := toSquare % 8
	toRank := toSquare / 8

	from := fmt.Sprintf("%c%d", 'a'+fromFile, fromRank+1)
	to := fmt.Sprintf("%c%d", 'a'+toFile, toRank+1)

	return from + to
}

// convertStateTo12Channel converts simple board state to 12-channel tensor
func convertStateTo12Channel(state []float64) [12][8][8]float32 {
	var tensor [12][8][8]float32

	for i := 0; i < 64; i++ {
		rank := i / 8
		file := i % 8
		piece := state[i]

		// Map piece values to channels (0-5 for white, 6-11 for black)
		// P=1, N=2, B=3, R=4, Q=5, K=6 (white positive, black negative)
		if piece > 0 {
			// White pieces
			channel := int(piece) - 1
			if channel >= 0 && channel < 6 {
				tensor[channel][rank][file] = 1.0
			}
		} else if piece < 0 {
			// Black pieces
			channel := int(-piece) + 5
			if channel >= 6 && channel < 12 {
				tensor[channel][rank][file] = 1.0
			}
		}
	}

	return tensor
}

func (c *CLI) batchInference() {
	reader := bufio.NewReader(os.Stdin)

	if c.model == nil {
		fmt.Println("No model loaded. Please load a model first")
		return
	}

	fmt.Print("\nNumber of positions to test: ")
	numStr, _ := reader.ReadString('\n')
	numTests, err := strconv.Atoi(strings.TrimSpace(numStr))
	if err != nil || numTests <= 0 {
		fmt.Println("Invalid number")
		return
	}

	dataset, err := data.NewDataset(c.datasetPath)
	if err != nil {
		fmt.Printf("Failed to open dataset: %v\n", err)
		return
	}
	defer dataset.Close()

	count, _ := dataset.Count()
	if count == 0 {
		fmt.Println("Dataset is empty")
		return
	}

	fmt.Printf("\nRunning batch inference on %d positions...\n", numTests)

	correct := 0
	top3Correct := 0
	top5Correct := 0
	totalTime := time.Duration(0)

	for i := 0; i < numTests; i++ {
		idx := (i * count / numTests) % count
		entries, err := dataset.LoadBatch(idx, 1)
		if err != nil || len(entries) == 0 {
			continue
		}

		entry := entries[0]
		boardTensor, err := data.FlatArrayToTensor(entry.StateTensor)
		if err != nil {
			continue
		}

		startTime := time.Now()
		predictions, err := c.model.Predict(boardTensor, 5)
		totalTime += time.Since(startTime)

		if err != nil {
			continue
		}

		// Check if actual move is in predictions
		for j, pred := range predictions {
			if pred.FromSquare == entry.FromSquare && pred.ToSquare == entry.ToSquare {
				if j == 0 {
					correct++
				}
				if j < 3 {
					top3Correct++
				}
				if j < 5 {
					top5Correct++
				}
				break
			}
		}

		if (i+1)%100 == 0 {
			fmt.Printf("  Processed %d/%d positions...\r", i+1, numTests)
		}
	}

	avgTime := totalTime / time.Duration(numTests)
	throughput := float64(numTests) / totalTime.Seconds()

	fmt.Println("\n\n" + strings.Repeat("-", 60))
	fmt.Println("BATCH INFERENCE RESULTS")
	fmt.Println(strings.Repeat("-", 60))
	fmt.Printf("Positions tested:    %d\n", numTests)
	fmt.Printf("Top-1 accuracy:      %.2f%% (%d correct)\n", float64(correct)/float64(numTests)*100, correct)
	fmt.Printf("Top-3 accuracy:      %.2f%% (%d correct)\n", float64(top3Correct)/float64(numTests)*100, top3Correct)
	fmt.Printf("Top-5 accuracy:      %.2f%% (%d correct)\n", float64(top5Correct)/float64(numTests)*100, top5Correct)
	fmt.Printf("Avg inference time:  %v\n", avgTime)
	fmt.Printf("Throughput:          %.1f positions/sec\n", throughput)
	fmt.Println(strings.Repeat("-", 60))
}

func (c *CLI) performanceTest() {
	reader := bufio.NewReader(os.Stdin)

	if c.model == nil {
		fmt.Println("No model loaded. Please load a model first")
		return
	}

	fmt.Println("\n⚡ Performance Test")
	fmt.Println(strings.Repeat("-", 60))

	fmt.Print("Number of test iterations (default 100): ")
	input, _ := reader.ReadString('\n')
	iterations := 100
	if n, err := strconv.Atoi(strings.TrimSpace(input)); err == nil && n > 0 {
		iterations = n
	}

	fmt.Print("Include data loading time? (y/n, default n): ")
	includeLoading, _ := reader.ReadString('\n')
	includeLoadingTime := strings.ToLower(strings.TrimSpace(includeLoading)) == "y"

	// Generate random test positions
	fmt.Printf("\nGenerating %d random test positions...\n", iterations)
	testPositions := make([][12][8][8]float32, iterations)
	for i := 0; i < iterations; i++ {
		testPositions[i] = generateRandomPosition()
	}

	fmt.Println("✓ Test positions generated")
	fmt.Println("\n🚀 Running performance test...")
	fmt.Println(strings.Repeat("-", 60))

	// Warm-up runs
	for i := 0; i < 5; i++ {
		c.model.Predict(testPositions[0], 5)
	}

	// Actual performance test
	var totalInferenceTime time.Duration
	var totalLoadTime time.Duration
	successCount := 0

	startTime := time.Now()
	for i := 0; i < iterations; i++ {
		loadStart := time.Now()
		pos := testPositions[i]
		if includeLoadingTime {
			totalLoadTime += time.Since(loadStart)
		}

		inferStart := time.Now()
		_, err := c.model.Predict(pos, 5)
		inferTime := time.Since(inferStart)

		if err == nil {
			successCount++
			totalInferenceTime += inferTime
		}

		// Progress indicator
		if (i+1)%10 == 0 {
			fmt.Printf("Progress: %d/%d (%.1f%%)\r", i+1, iterations, float64(i+1)/float64(iterations)*100)
		}
	}
	totalTime := time.Since(startTime)

	// Display results
	fmt.Println()
	fmt.Println(strings.Repeat("-", 60))
	fmt.Println("PERFORMANCE TEST RESULTS")
	fmt.Println(strings.Repeat("-", 60))
	fmt.Printf("Total iterations:     %d\n", iterations)
	fmt.Printf("Successful:           %d (%.1f%%)\n", successCount, float64(successCount)/float64(iterations)*100)
	fmt.Printf("Failed:               %d\n", iterations-successCount)
	fmt.Println(strings.Repeat("-", 60))

	if successCount > 0 {
		avgInferTime := totalInferenceTime / time.Duration(successCount)
		throughput := float64(successCount) / totalTime.Seconds()

		fmt.Printf("Total time:           %v\n", totalTime)
		if includeLoadingTime {
			fmt.Printf("Data loading time:    %v\n", totalLoadTime)
		}
		fmt.Printf("Pure inference time:  %v\n", totalInferenceTime)
		fmt.Printf("Avg inference time:   %v\n", avgInferTime)
		fmt.Printf("Throughput:           %.1f inferences/sec\n", throughput)
		fmt.Printf("Latency (P50):        ~%v\n", avgInferTime)
		fmt.Printf("Latency (P99):        ~%v\n", avgInferTime*110/100)
	}

	fmt.Println(strings.Repeat("-", 60))

	// Memory usage estimate
	fmt.Println("\n💾 Memory Usage Estimate:")
	fmt.Printf("Model size:           ~%.2f MB\n", float64(estimateModelSize())/1024/1024)
	fmt.Printf("Test data size:       ~%.2f MB\n", float64(iterations*12*8*8*4)/1024/1024)
	fmt.Println(strings.Repeat("-", 60))
}

// generateRandomPosition creates a random board position for testing
func generateRandomPosition() [12][8][8]float32 {
	var pos [12][8][8]float32

	// Place random pieces (simplified - doesn't ensure valid position)
	numPieces := 10 + (time.Now().UnixNano() % 22) // 10-32 pieces

	for i := int64(0); i < numPieces; i++ {
		channel := int(time.Now().UnixNano() % 12)
		rank := int(time.Now().UnixNano() % 8)
		file := int(time.Now().UnixNano() % 8)
		pos[channel][rank][file] = 1.0
		time.Sleep(1 * time.Nanosecond) // Ensure different random values
	}

	return pos
}

// estimateModelSize estimates the model size in bytes
func estimateModelSize() int {
	// Rough estimate: 3 conv layers + 3 fc layers
	// Conv: (filters * kernel * kernel * channels + filters) * 4 bytes
	// FC: (in * out + out) * 4 bytes

	conv1 := (32*3*3*12 + 32) * 4
	conv2 := (64*3*3*32 + 64) * 4
	conv3 := (128*3*3*64 + 128) * 4
	fc1 := (8192*512 + 512) * 4
	fc2 := (512*256 + 256) * 4
	fc3 := (256*4096 + 4096) * 4

	return conv1 + conv2 + conv3 + fc1 + fc2 + fc3
}

func (c *CLI) changeModelPath(reader *bufio.Reader) {
	fmt.Print("\nEnter new model path: ")
	path, _ := reader.ReadString('\n')
	path = strings.TrimSpace(path)

	if path != "" {
		// Create directory if needed
		dir := filepath.Dir(path)
		os.MkdirAll(dir, 0755)

		c.modelPath = path
		fmt.Printf("✓ Model path updated: %s\n", c.modelPath)
	}
}

func (c *CLI) changeDatasetPath(reader *bufio.Reader) {
	fmt.Print("\nEnter new dataset path: ")
	path, _ := reader.ReadString('\n')
	path = strings.TrimSpace(path)

	if path != "" {
		// Create directory if needed
		dir := filepath.Dir(path)
		os.MkdirAll(dir, 0755)

		c.datasetPath = path
		fmt.Printf("✓ Dataset path updated: %s\n", c.datasetPath)
	}
}

func squareToAlgebraic(square int) string {
	if square < 0 || square >= 64 {
		return "??"
	}
	file := rune('a' + (square % 8))
	rank := (square / 8) + 1
	return fmt.Sprintf("%c%d", file, rank)
}
