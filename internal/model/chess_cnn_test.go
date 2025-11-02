package model

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/thyrook/partner/internal/data"
)

func TestNewChessCNN(t *testing.T) {
	model, err := NewChessCNN()
	if err != nil {
		t.Fatalf("Failed to create model: %v", err)
	}
	defer model.Close()

	if model.g == nil {
		t.Error("Graph is nil")
	}

	if model.input == nil {
		t.Error("Input node is nil")
	}

	if model.output == nil {
		t.Error("Output node is nil")
	}

	// Check learnables
	learnables := model.Learnables()
	expectedCount := 10 // 5 layers * 2 (weight + bias)
	if len(learnables) != expectedCount {
		t.Errorf("Expected %d learnables, got %d", expectedCount, len(learnables))
	}
}

func TestPredictForwardPass(t *testing.T) {
	model, err := NewChessCNN()
	if err != nil {
		t.Fatalf("Failed to create model: %v", err)
	}
	defer model.Close()

	// Create a dummy board tensor (starting position)
	var boardTensor [12][8][8]float32

	// Add some white pawns
	for i := 0; i < 8; i++ {
		boardTensor[0][6][i] = 1.0 // White pawns on rank 2
	}

	// Add some black pawns
	for i := 0; i < 8; i++ {
		boardTensor[6][1][i] = 1.0 // Black pawns on rank 7
	}

	// Run prediction
	predictions, err := model.Predict(boardTensor, 3)
	if err != nil {
		t.Fatalf("Prediction failed: %v", err)
	}

	if len(predictions) != 3 {
		t.Errorf("Expected 3 predictions, got %d", len(predictions))
	}

	// Check that probabilities sum to ~1.0 (with some tolerance)
	totalProb := 0.0
	for _, pred := range predictions {
		totalProb += pred.Probability

		if pred.FromSquare < 0 || pred.FromSquare >= 64 {
			t.Errorf("Invalid from square: %d", pred.FromSquare)
		}

		if pred.ToSquare < 0 || pred.ToSquare >= 64 {
			t.Errorf("Invalid to square: %d", pred.ToSquare)
		}

		if pred.Probability < 0.0 || pred.Probability > 1.0 {
			t.Errorf("Invalid probability: %f", pred.Probability)
		}
	}

	// Top 3 should be a small fraction of total probability
	if totalProb <= 0.0 || totalProb > 1.0 {
		t.Errorf("Sum of top 3 probabilities is unreasonable: %f", totalProb)
	}
}

func TestSaveLoadModel(t *testing.T) {
	// Create temp directory
	tmpDir := t.TempDir()
	modelPath := filepath.Join(tmpDir, "test_model.gob")

	// Create and save model
	model1, err := NewChessCNN()
	if err != nil {
		t.Fatalf("Failed to create model: %v", err)
	}
	defer model1.Close()

	// Run a forward pass to initialize weights
	var boardTensor [12][8][8]float32
	boardTensor[0][6][4] = 1.0 // Add a piece

	_, err = model1.Predict(boardTensor, 1)
	if err != nil {
		t.Fatalf("Initial prediction failed: %v", err)
	}

	// Save model
	if err := model1.SaveModel(modelPath); err != nil {
		t.Fatalf("Failed to save model: %v", err)
	}

	// Check file exists
	if _, err := os.Stat(modelPath); os.IsNotExist(err) {
		t.Fatal("Model file was not created")
	}

	// Create new model and load weights
	model2, err := NewChessCNN()
	if err != nil {
		t.Fatalf("Failed to create second model: %v", err)
	}
	defer model2.Close()

	if err := model2.LoadModel(modelPath); err != nil {
		t.Fatalf("Failed to load model: %v", err)
	}

	// Compare predictions from both models (should be identical)
	pred1, err := model1.Predict(boardTensor, 3)
	if err != nil {
		t.Fatalf("First model prediction failed: %v", err)
	}

	pred2, err := model2.Predict(boardTensor, 3)
	if err != nil {
		t.Fatalf("Second model prediction failed: %v", err)
	}

	// Check predictions match
	for i := 0; i < len(pred1); i++ {
		if pred1[i].MoveIndex != pred2[i].MoveIndex {
			t.Errorf("Move index mismatch at position %d: %d vs %d",
				i, pred1[i].MoveIndex, pred2[i].MoveIndex)
		}

		probDiff := pred1[i].Probability - pred2[i].Probability
		if probDiff < 0 {
			probDiff = -probDiff
		}
		if probDiff > 1e-6 {
			t.Errorf("Probability mismatch at position %d: %f vs %f",
				i, pred1[i].Probability, pred2[i].Probability)
		}
	}
}

func TestConvertMoveToTarget(t *testing.T) {
	tests := []struct {
		name       string
		fromSquare int
		toSquare   int
		expectErr  bool
	}{
		{"Valid move", 12, 28, false},
		{"Corner to corner", 0, 63, false},
		{"Same square", 10, 10, false},
		{"Invalid from", -1, 28, true},
		{"Invalid to", 12, 64, true},
		{"Both invalid", -1, 100, true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			target, err := ConvertMoveToTarget(tt.fromSquare, tt.toSquare)

			if tt.expectErr {
				if err == nil {
					t.Error("Expected error but got nil")
				}
				return
			}

			if err != nil {
				t.Fatalf("Unexpected error: %v", err)
			}

			// Check target vector
			if len(target) != 4096 {
				t.Errorf("Expected target length 4096, got %d", len(target))
			}

			// Check that exactly one element is 1.0
			oneCount := 0
			moveIndex := tt.fromSquare*64 + tt.toSquare
			for i, v := range target {
				if v == 1.0 {
					oneCount++
					if i != moveIndex {
						t.Errorf("Expected 1.0 at index %d, but found at %d", moveIndex, i)
					}
				} else if v != 0.0 {
					t.Errorf("Expected 0.0 or 1.0, got %f at index %d", v, i)
				}
			}

			if oneCount != 1 {
				t.Errorf("Expected exactly 1 element to be 1.0, got %d", oneCount)
			}
		})
	}
}

func TestClipGradients(t *testing.T) {
	model, err := NewChessCNN()
	if err != nil {
		t.Fatalf("Failed to create model: %v", err)
	}
	defer model.Close()

	// Just test that the function doesn't crash
	// (gradients will be nil initially, but function should handle that)
	err = ClipGradients(model.Learnables(), 5.0)
	if err != nil {
		t.Errorf("ClipGradients failed: %v", err)
	}
}

func TestTrainerCreation(t *testing.T) {
	config := DefaultTrainingConfig()
	config.Epochs = 2
	config.BatchSize = 4

	trainer, err := NewTrainer(config)
	if err != nil {
		t.Fatalf("Failed to create trainer: %v", err)
	}

	if trainer == nil {
		t.Fatal("Trainer is nil")
	}

	model := trainer.GetModel()
	if model == nil {
		t.Error("Trainer model is nil")
	}
	defer model.Close()

	if trainer.solver == nil {
		t.Error("Trainer solver is nil")
	}
}

func TestTrainingWithSyntheticData(t *testing.T) {
	// Skip if too slow
	if testing.Short() {
		t.Skip("Skipping training test in short mode")
	}

	// Create temp directory for dataset and model
	tmpDir := t.TempDir()
	datasetPath := filepath.Join(tmpDir, "test_dataset.db")
	modelPath := filepath.Join(tmpDir, "test_model.gob")

	// Create synthetic dataset
	dataset, err := data.NewDataset(datasetPath)
	if err != nil {
		t.Fatalf("Failed to create dataset: %v", err)
	}
	defer dataset.Close()

	// Add synthetic training samples
	numSamples := 10
	for i := 0; i < numSamples; i++ {
		// Create a simple board state
		var tensor [12][8][8]float32
		tensor[0][6][i%8] = 1.0 // White pawn

		flat := data.TensorToFlatArray(tensor)

		entry := &data.DataEntry{
			StateTensor: flat,
			FromSquare:  i % 64,
			ToSquare:    (i + 1) % 64,
			GameID:      "synthetic",
			MoveNumber:  i,
		}

		if err := dataset.Add(entry); err != nil {
			t.Fatalf("Failed to add entry: %v", err)
		}
	}

	// Create trainer with minimal config
	config := &TrainingConfig{
		Epochs:          2,
		BatchSize:       4,
		LearningRate:    0.001,
		LRDecayRate:     0.95,
		LRDecaySteps:    10,
		GradientClipMax: 5.0,
		Verbose:         false,
		SaveInterval:    1,
		SavePath:        modelPath,
	}

	trainer, err := NewTrainer(config)
	if err != nil {
		t.Fatalf("Failed to create trainer: %v", err)
	}
	
	model := trainer.GetModel()
	defer model.Close()

	// Train
	err = trainer.Train(dataset)
	if err != nil {
		t.Fatalf("Training failed: %v", err)
	}

	// Check metrics were recorded
	metrics := trainer.GetMetrics()
	if len(metrics) != config.Epochs {
		t.Errorf("Expected %d metrics entries, got %d", config.Epochs, len(metrics))
	}

	// Check model was saved
	if _, err := os.Stat(modelPath); os.IsNotExist(err) {
		t.Error("Model file was not saved")
	}

	// Verify we can load the saved model
	loadedModel, err := NewChessCNN()
	if err != nil {
		t.Fatalf("Failed to create model for loading: %v", err)
	}
	defer loadedModel.Close()

	if err := loadedModel.LoadModel(modelPath); err != nil {
		t.Errorf("Failed to load saved model: %v", err)
	}
}

func TestGetTopKPredictions(t *testing.T) {
	// Create sample probabilities
	probs := make([]float64, 100)
	probs[10] = 0.5
	probs[20] = 0.3
	probs[30] = 0.2

	predictions := getTopKPredictions(probs, 3)

	if len(predictions) != 3 {
		t.Errorf("Expected 3 predictions, got %d", len(predictions))
	}

	// Check ordering (should be descending by probability)
	if predictions[0].Probability < predictions[1].Probability {
		t.Error("Predictions not sorted by probability")
	}

	if predictions[1].Probability < predictions[2].Probability {
		t.Error("Predictions not sorted by probability")
	}

	// Check top prediction
	if predictions[0].MoveIndex != 10 || predictions[0].Probability != 0.5 {
		t.Errorf("Top prediction incorrect: index=%d, prob=%f",
			predictions[0].MoveIndex, predictions[0].Probability)
	}
}

func TestModelMetadata(t *testing.T) {
	metadata := ModelMetadata{
		Version:     "1.0",
		ModelType:   "ChessCNN",
		InputShape:  []int{12, 8, 8},
		OutputShape: []int{4096},
	}

	if metadata.Version != "1.0" {
		t.Errorf("Expected version 1.0, got %s", metadata.Version)
	}

	if metadata.ModelType != "ChessCNN" {
		t.Errorf("Expected ModelType ChessCNN, got %s", metadata.ModelType)
	}

	if len(metadata.InputShape) != 3 {
		t.Errorf("Expected InputShape length 3, got %d", len(metadata.InputShape))
	}

	if len(metadata.OutputShape) != 1 {
		t.Errorf("Expected OutputShape length 1, got %d", len(metadata.OutputShape))
	}
}

func TestInferenceModelCreation(t *testing.T) {
	// Create temp directory
	tmpDir := t.TempDir()
	modelPath := filepath.Join(tmpDir, "test_model.gob")

	// Create and save a model
	trainModel, err := NewChessCNNWithBatchSize(8) // Train with batch size 8
	if err != nil {
		t.Fatalf("Failed to create training model: %v", err)
	}
	defer trainModel.Close()

	// Run forward pass to initialize
	var boardTensor [12][8][8]float32
	boardTensor[0][6][4] = 1.0

	if err := trainModel.SaveModel(modelPath); err != nil {
		t.Fatalf("Failed to save model: %v", err)
	}

	// Create inference model with batch size 1
	inferenceModel, err := NewChessCNNForInference(modelPath)
	if err != nil {
		t.Fatalf("Failed to create inference model: %v", err)
	}
	defer inferenceModel.Close()

	// Test prediction works with batch size 1
	predictions, err := inferenceModel.Predict(boardTensor, 3)
	if err != nil {
		t.Fatalf("Inference prediction failed: %v", err)
	}

	if len(predictions) != 3 {
		t.Errorf("Expected 3 predictions, got %d", len(predictions))
	}
}
