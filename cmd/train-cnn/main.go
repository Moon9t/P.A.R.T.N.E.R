package main

import (
	"flag"
	"fmt"
	"os"

	"github.com/thyrook/partner/internal/data"
	"github.com/thyrook/partner/internal/model"
)

func main() {
	// Command-line flags
	datasetPath := flag.String("dataset", "data/chess_dataset.db", "Path to training dataset")
	modelPath := flag.String("model", "models/chess_cnn.gob", "Path to save/load model")
	epochs := flag.Int("epochs", 10, "Number of training epochs")
	batchSize := flag.Int("batch-size", 64, "Batch size for training")
	learningRate := flag.Float64("lr", 0.001, "Learning rate")
	loadModel := flag.Bool("load", false, "Load existing model before training")
	testMode := flag.Bool("test", false, "Test mode: just run inference on a sample")

	flag.Parse()

	fmt.Println("Chess CNN Training Tool")
	fmt.Println("=" + "=" + "=" + "=" + "=" + "=" + "=" + "=" + "=" + "=" + "=" + "=" + "=" + "=" + "=" + "=")
	fmt.Println()

	// Test mode: just create model and run inference
	if *testMode {
		runTestMode(*modelPath, *loadModel)
		return
	}

	// Open dataset
	fmt.Printf("Loading dataset from: %s\n", *datasetPath)
	dataset, err := data.NewDataset(*datasetPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Failed to open dataset: %v\n", err)
		os.Exit(1)
	}
	defer dataset.Close()

	// Get dataset stats
	stats, err := dataset.GetStats()
	if err != nil {
		fmt.Fprintf(os.Stderr, "Failed to get dataset stats: %v\n", err)
		os.Exit(1)
	}

	fmt.Printf("Dataset entries: %d\n", stats.TotalEntries)
	fmt.Printf("Dataset size: %.2f MB\n", float64(stats.FileSize)/1024/1024)
	fmt.Println()

	if stats.TotalEntries == 0 {
		fmt.Fprintf(os.Stderr, "Dataset is empty. Please ingest data first.\n")
		os.Exit(1)
	}

	// Configure training
	config := &model.TrainingConfig{
		Epochs:          *epochs,
		BatchSize:       *batchSize,
		LearningRate:    *learningRate,
		LRDecayRate:     0.95,
		LRDecaySteps:    2,
		GradientClipMax: 5.0,
		Verbose:         true,
		SaveInterval:    2,
		SavePath:        *modelPath,
	}

	fmt.Println()
	fmt.Println("Training Configuration:")
	fmt.Printf("  Epochs:          %d\n", config.Epochs)
	fmt.Printf("  Batch size:      %d\n", config.BatchSize)
	fmt.Printf("  Learning rate:   %.6f\n", config.LearningRate)
	fmt.Printf("  LR decay:        %.2f every %d epochs\n", config.LRDecayRate, config.LRDecaySteps)
	fmt.Printf("  Gradient clip:   %.1f\n", config.GradientClipMax)
	fmt.Printf("  Save interval:   %d epochs\n", config.SaveInterval)
	fmt.Printf("  Model path:      %s\n", config.SavePath)
	fmt.Println()

	// Create trainer (creates model with batch size from config)
	fmt.Println("Creating Chess CNN model and trainer...")
	trainer, err := model.NewTrainer(config)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Failed to create trainer: %v\n", err)
		os.Exit(1)
	}

	cnnModel := trainer.GetModel()
	defer cnnModel.Close()

	// Load existing model if requested
	if *loadModel {
		if _, err := os.Stat(*modelPath); err == nil {
			fmt.Printf("Loading existing model from: %s\n", *modelPath)
			if err := cnnModel.LoadModel(*modelPath); err != nil {
				fmt.Fprintf(os.Stderr, "Failed to load model: %v\n", err)
				os.Exit(1)
			}
			fmt.Println("✓ Model loaded successfully")
		} else {
			fmt.Println("No existing model found, starting fresh")
		}
	}

	// Train with progress callback
	fmt.Println("Starting training...")
	fmt.Println("=" + "=" + "=" + "=" + "=" + "=" + "=" + "=" + "=" + "=" + "=" + "=" + "=" + "=" + "=" + "=")
	fmt.Println()

	err = trainer.TrainWithCallback(dataset, func(metrics model.TrainingMetrics) {
		fmt.Printf("Epoch %d/%d - Loss: %.4f, Accuracy: %.2f%%, LR: %.6f, Time: %v\n",
			metrics.Epoch, config.Epochs,
			metrics.Loss,
			metrics.Accuracy*100,
			metrics.LearningRate,
			metrics.Duration)
	})

	if err != nil {
		fmt.Fprintf(os.Stderr, "\nTraining failed: %v\n", err)
		os.Exit(1)
	}

	fmt.Println()
	fmt.Println("=" + "=" + "=" + "=" + "=" + "=" + "=" + "=" + "=" + "=" + "=" + "=" + "=" + "=" + "=" + "=")
	fmt.Println("Training completed successfully!")
	fmt.Printf("Model saved to: %s\n", *modelPath)

	// Show final metrics
	metrics := trainer.GetMetrics()
	if len(metrics) > 0 {
		final := metrics[len(metrics)-1]
		fmt.Println()
		fmt.Println("Final Results:")
		fmt.Printf("  Final Loss:     %.4f\n", final.Loss)
		fmt.Printf("  Final Accuracy: %.2f%%\n", final.Accuracy*100)
		fmt.Printf("  Total samples:  %d\n", final.SamplesSeen)
	}

	// Test inference with a separate inference model
	fmt.Println()
	fmt.Println("Running inference test...")
	testInferenceWithCheckpoint(*modelPath, dataset)
}

func runTestMode(modelPath string, loadModel bool) {
	fmt.Println("Test Mode: Model compilation and inference check")
	fmt.Println()

	// Create model
	fmt.Println("Creating Chess CNN model...")
	cnnModel, err := model.NewChessCNN()
	if err != nil {
		fmt.Fprintf(os.Stderr, "Failed to create model: %v\n", err)
		os.Exit(1)
	}
	defer cnnModel.Close()

	fmt.Println("✓ Model created successfully")

	// Load model if requested
	if loadModel {
		if _, err := os.Stat(modelPath); err == nil {
			fmt.Printf("Loading model from: %s\n", modelPath)
			if err := cnnModel.LoadModel(modelPath); err != nil {
				fmt.Fprintf(os.Stderr, "Failed to load model: %v\n", err)
				os.Exit(1)
			}
			fmt.Println("✓ Model loaded successfully")
		} else {
			fmt.Println("⚠ No model found, using random initialization")
		}
	}

	// Create a test board state
	fmt.Println()
	fmt.Println("Running forward pass test...")

	var boardTensor [12][8][8]float32

	// Add starting position pieces
	for i := 0; i < 8; i++ {
		boardTensor[0][6][i] = 1.0 // White pawns
		boardTensor[6][1][i] = 1.0 // Black pawns
	}

	// White rooks
	boardTensor[3][7][0] = 1.0
	boardTensor[3][7][7] = 1.0

	// Black rooks
	boardTensor[9][0][0] = 1.0
	boardTensor[9][0][7] = 1.0

	// Run prediction
	predictions, err := cnnModel.Predict(boardTensor, 5)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Prediction failed: %v\n", err)
		os.Exit(1)
	}

	fmt.Println("✓ Forward pass successful")
	fmt.Println()
	fmt.Println("Top 5 predictions:")
	for i, pred := range predictions {
		fromFile := rune('a' + (pred.FromSquare % 8))
		fromRank := (pred.FromSquare / 8) + 1
		toFile := rune('a' + (pred.ToSquare % 8))
		toRank := (pred.ToSquare / 8) + 1

		fmt.Printf("  %d. %c%d → %c%d  (prob: %.4f%%)\n",
			i+1, fromFile, fromRank, toFile, toRank, pred.Probability*100)
	}

	fmt.Println()
	fmt.Println("✓ All tests passed!")
}

func testInferenceWithCheckpoint(modelPath string, dataset *data.Dataset) {
	// Check if model file exists
	if _, err := os.Stat(modelPath); os.IsNotExist(err) {
		fmt.Println("⚠ Model checkpoint not found, skipping inference test")
		return
	}

	// Create inference model with batch size 1 and load checkpoint
	fmt.Println("Loading model for inference...")
	inferenceModel, err := model.NewChessCNNForInference(modelPath)
	if err != nil {
		fmt.Printf("⚠ Failed to create inference model: %v\n", err)
		return
	}
	defer inferenceModel.Close()

	// Load one sample from dataset
	entries, err := dataset.LoadBatch(0, 1)
	if err != nil || len(entries) == 0 {
		fmt.Println("⚠ Could not load sample from dataset")
		return
	}

	entry := entries[0]

	// Convert to tensor
	tensor, err := data.FlatArrayToTensor(entry.StateTensor)
	if err != nil {
		fmt.Printf("⚠ Failed to convert sample: %v\n", err)
		return
	}

	// Run prediction
	predictions, err := inferenceModel.Predict(tensor, 3)
	if err != nil {
		fmt.Printf("⚠ Inference failed: %v\n", err)
		return
	}

	fmt.Printf("Sample from game: %s, move #%d\n", entry.GameID, entry.MoveNumber)
	fmt.Printf("Actual move: %d → %d\n", entry.FromSquare, entry.ToSquare)
	fmt.Println()
	fmt.Println("Top 3 predictions:")

	for i, pred := range predictions {
		fromFile := rune('a' + (pred.FromSquare % 8))
		fromRank := (pred.FromSquare / 8) + 1
		toFile := rune('a' + (pred.ToSquare % 8))
		toRank := (pred.ToSquare / 8) + 1

		marker := "  "
		if pred.FromSquare == entry.FromSquare && pred.ToSquare == entry.ToSquare {
			marker = "✓ "
		}

		fmt.Printf("%s%d. %c%d → %c%d  (prob: %.4f%%)\n",
			marker, i+1, fromFile, fromRank, toFile, toRank, pred.Probability*100)
	}

	fmt.Println()
	fmt.Println("✓ Inference test completed successfully!")
}
