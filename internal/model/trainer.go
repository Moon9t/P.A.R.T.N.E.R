package model

import (
	"fmt"
	"math"
	"time"

	"github.com/thyrook/partner/internal/data"
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

// TrainingConfig holds training hyperparameters
type TrainingConfig struct {
	Epochs          int
	BatchSize       int
	LearningRate    float64
	LRDecayRate     float64 // Learning rate decay per epoch
	LRDecaySteps    int     // Decay every N epochs
	GradientClipMax float64
	Verbose         bool
	SaveInterval    int // Save model every N epochs
	SavePath        string
}

// DefaultTrainingConfig returns default training configuration
func DefaultTrainingConfig() *TrainingConfig {
	return &TrainingConfig{
		Epochs:          10,
		BatchSize:       64,
		LearningRate:    0.001,
		LRDecayRate:     0.95,
		LRDecaySteps:    2,
		GradientClipMax: 5.0,
		Verbose:         true,
		SaveInterval:    5,
		SavePath:        "models/chess_cnn.gob",
	}
}

// TrainingMetrics tracks training progress
type TrainingMetrics struct {
	Epoch        int
	Loss         float64
	Accuracy     float64
	LearningRate float64
	Duration     time.Duration
	SamplesSeen  int
}

// Trainer manages the training process
type Trainer struct {
	model      *ChessCNN
	config     *TrainingConfig
	solver     gorgonia.Solver
	metrics    []TrainingMetrics
	targetNode *gorgonia.Node
	lossNode   *gorgonia.Node
}

// NewTrainer creates a new trainer with a model that supports the specified batch size
func NewTrainer(config *TrainingConfig) (*Trainer, error) {
	if config == nil {
		config = DefaultTrainingConfig()
	}

	// Create model with batch size from config
	model, err := NewChessCNNWithBatchSize(config.BatchSize)
	if err != nil {
		return nil, fmt.Errorf("failed to create model: %w", err)
	}

	// Create Adam solver
	solver := gorgonia.NewAdamSolver(
		gorgonia.WithLearnRate(config.LearningRate),
		gorgonia.WithBatchSize(float64(config.BatchSize)),
		gorgonia.WithClip(config.GradientClipMax),
	)

	// Create target node for training
	targetNode := gorgonia.NewMatrix(
		model.g,
		tensor.Float64,
		gorgonia.WithShape(config.BatchSize, 4096),
		gorgonia.WithName("target"),
	)

	// Create loss node
	lossNode, err := model.ComputeLoss(targetNode)
	if err != nil {
		return nil, fmt.Errorf("failed to create loss node: %w", err)
	}

	// Compute gradients
	if _, err := gorgonia.Grad(lossNode, model.Learnables()...); err != nil {
		return nil, fmt.Errorf("failed to compute gradients: %w", err)
	}

	// Recreate VM now that we have the full graph including loss and gradients
	model.vm.Close()
	model.vm = gorgonia.NewTapeMachine(model.g)

	return &Trainer{
		model:      model,
		config:     config,
		solver:     solver,
		metrics:    make([]TrainingMetrics, 0),
		targetNode: targetNode,
		lossNode:   lossNode,
	}, nil
}

// Train trains the model on the dataset
func (t *Trainer) Train(dataset *data.Dataset) error {
	if dataset == nil {
		return fmt.Errorf("dataset is nil")
	}

	// Get total samples
	totalSamples, err := dataset.Count()
	if err != nil {
		return fmt.Errorf("failed to get dataset count: %w", err)
	}

	if totalSamples == 0 {
		return fmt.Errorf("dataset is empty")
	}

	if t.config.Verbose {
		fmt.Printf("Starting training with %d samples\n", totalSamples)
		fmt.Printf("Epochs: %d, Batch size: %d, Learning rate: %.6f\n",
			t.config.Epochs, t.config.BatchSize, t.config.LearningRate)
		fmt.Println()
	}

	// Training loop
	for epoch := 0; epoch < t.config.Epochs; epoch++ {
		startTime := time.Now()

		// Apply learning rate decay
		if epoch > 0 && epoch%t.config.LRDecaySteps == 0 {
			newLR := t.config.LearningRate * math.Pow(t.config.LRDecayRate, float64(epoch/t.config.LRDecaySteps))
			t.solver = gorgonia.NewAdamSolver(
				gorgonia.WithLearnRate(newLR),
				gorgonia.WithBatchSize(float64(t.config.BatchSize)),
				gorgonia.WithClip(t.config.GradientClipMax),
			)
			if t.config.Verbose {
				fmt.Printf("Learning rate decayed to: %.6f\n", newLR)
			}
		}

		// Train one epoch
		epochLoss, accuracy, samplesSeen, err := t.trainEpoch(dataset, totalSamples)
		if err != nil {
			return fmt.Errorf("epoch %d failed: %w", epoch+1, err)
		}

		duration := time.Since(startTime)

		// Record metrics
		metrics := TrainingMetrics{
			Epoch:        epoch + 1,
			Loss:         epochLoss,
			Accuracy:     accuracy,
			LearningRate: t.config.LearningRate,
			Duration:     duration,
			SamplesSeen:  samplesSeen,
		}
		t.metrics = append(t.metrics, metrics)

		// Print progress
		if t.config.Verbose {
			fmt.Printf("Epoch %d/%d - Loss: %.4f, Accuracy: %.2f%%, Time: %v\n",
				epoch+1, t.config.Epochs, epochLoss, accuracy*100, duration)
		}

		// Save checkpoint
		if t.config.SaveInterval > 0 && (epoch+1)%t.config.SaveInterval == 0 {
			if err := t.model.SaveModel(t.config.SavePath); err != nil {
				fmt.Printf("Warning: Failed to save checkpoint: %v\n", err)
			} else if t.config.Verbose {
				fmt.Printf("Checkpoint saved to %s\n", t.config.SavePath)
			}
		}
	}

	// Final save
	if t.config.SavePath != "" {
		if err := t.model.SaveModel(t.config.SavePath); err != nil {
			return fmt.Errorf("failed to save final model: %w", err)
		}
		if t.config.Verbose {
			fmt.Printf("\nModel saved to %s\n", t.config.SavePath)
		}
	}

	return nil
}

// trainEpoch trains for one epoch
func (t *Trainer) trainEpoch(dataset *data.Dataset, totalSamples int) (float64, float64, int, error) {
	batchSize := t.config.BatchSize
	numBatches := (totalSamples + batchSize - 1) / batchSize

	totalLoss := 0.0
	correctPredictions := 0
	samplesSeen := 0

	for batchIdx := 0; batchIdx < numBatches; batchIdx++ {
		offset := batchIdx * batchSize

		// Load batch from dataset
		entries, err := dataset.LoadBatch(offset, batchSize)
		if err != nil {
			return 0, 0, 0, fmt.Errorf("failed to load batch: %w", err)
		}

		if len(entries) == 0 {
			break
		}

		// Train on batch
		batchLoss, batchCorrect, err := t.trainBatch(entries)
		if err != nil {
			return 0, 0, 0, fmt.Errorf("batch %d failed: %w", batchIdx, err)
		}

		totalLoss += batchLoss
		correctPredictions += batchCorrect
		samplesSeen += len(entries)
	}

	avgLoss := totalLoss / float64(numBatches)
	accuracy := float64(correctPredictions) / float64(samplesSeen)

	return avgLoss, accuracy, samplesSeen, nil
}

// trainBatch trains on a batch of samples together
func (t *Trainer) trainBatch(entries []*data.DataEntry) (float64, int, error) {
	if len(entries) == 0 {
		return 0, 0, nil
	}

	batchSize := len(entries)

	// If batch size doesn't match model batch size, pad or truncate
	if batchSize != t.config.BatchSize {
		// For the last batch, we can handle it differently
		// For now, process samples one by one if batch is smaller
		if batchSize < t.config.BatchSize {
			return t.trainBatchSmall(entries)
		}
		// Truncate if larger (shouldn't happen with proper batching)
		entries = entries[:t.config.BatchSize]
		batchSize = t.config.BatchSize
	}

	// Prepare batch tensors
	inputData := make([]float64, batchSize*12*8*8)
	targetData := make([]float64, batchSize*4096)

	// Fill batch data
	for i, entry := range entries {
		// Convert state tensor
		boardTensor, err := data.FlatArrayToTensor(entry.StateTensor)
		if err != nil {
			return 0, 0, fmt.Errorf("failed to convert entry %d: %w", i, err)
		}

		// Copy board data to batch input
		offset := i * 12 * 8 * 8
		idx := 0
		for c := 0; c < 12; c++ {
			for r := 0; r < 8; r++ {
				for f := 0; f < 8; f++ {
					inputData[offset+idx] = float64(boardTensor[c][r][f])
					idx++
				}
			}
		}

		// Create target for this sample
		targetVec, err := ConvertMoveToTarget(entry.FromSquare, entry.ToSquare)
		if err != nil {
			return 0, 0, fmt.Errorf("failed to create target for entry %d: %w", i, err)
		}

		// Copy target data to batch targets
		copy(targetData[i*4096:(i+1)*4096], targetVec)
	}

	// Create batch tensors
	inputTensor := tensor.New(
		tensor.WithShape(batchSize, 12, 8, 8),
		tensor.WithBacking(inputData),
	)

	targetTensor := tensor.New(
		tensor.WithShape(batchSize, 4096),
		tensor.WithBacking(targetData),
	)

	// Set input and target
	if err := gorgonia.Let(t.model.input, inputTensor); err != nil {
		return 0, 0, fmt.Errorf("failed to set input: %w", err)
	}

	if err := gorgonia.Let(t.targetNode, targetTensor); err != nil {
		return 0, 0, fmt.Errorf("failed to set target: %w", err)
	}

	// Run forward and backward pass
	if err := t.model.vm.RunAll(); err != nil {
		return 0, 0, fmt.Errorf("failed to run forward/backward: %w", err)
	}

	// Get loss value
	lossValue := t.lossNode.Value()
	if lossValue == nil {
		return 0, 0, fmt.Errorf("loss value is nil")
	}

	// Extract scalar loss value
	var avgLoss float64
	switch v := lossValue.Data().(type) {
	case float64:
		avgLoss = v
	case []float64:
		if len(v) > 0 {
			avgLoss = v[0]
		} else {
			return 0, 0, fmt.Errorf("loss value array is empty")
		}
	default:
		return 0, 0, fmt.Errorf("unexpected loss value type: %T", v)
	}

	// Update weights
	learnables := t.model.Learnables()
	valueGrads := make([]gorgonia.ValueGrad, len(learnables))
	for i, n := range learnables {
		valueGrads[i] = n
	}
	if err := t.solver.Step(valueGrads); err != nil {
		return 0, 0, fmt.Errorf("failed to update weights: %w", err)
	}

	// Reset VM for next batch
	t.model.vm.Reset()

	// Count correct predictions
	correctCount := 0
	outputValue := t.model.output.Value()
	if outputValue != nil {
		outputData := outputValue.Data().([]float64)

		for i, entry := range entries {
			// Get predicted move (argmax of output)
			maxIdx := 0
			maxProb := outputData[i*4096]
			for j := 1; j < 4096; j++ {
				if outputData[i*4096+j] > maxProb {
					maxProb = outputData[i*4096+j]
					maxIdx = j
				}
			}

			// Check if prediction is correct
			expectedIdx := entry.FromSquare*64 + entry.ToSquare
			if maxIdx == expectedIdx {
				correctCount++
			}
		}
	}

	return avgLoss, correctCount, nil
}

// trainBatchSmall handles batches smaller than the configured batch size
func (t *Trainer) trainBatchSmall(entries []*data.DataEntry) (float64, int, error) {
	// Pad the batch with zeros to match model batch size
	batchSize := t.config.BatchSize
	inputData := make([]float64, batchSize*12*8*8)
	targetData := make([]float64, batchSize*4096)

	actualSize := len(entries)

	// Fill only the actual entries
	for i, entry := range entries {
		boardTensor, err := data.FlatArrayToTensor(entry.StateTensor)
		if err != nil {
			return 0, 0, fmt.Errorf("failed to convert entry %d: %w", i, err)
		}

		offset := i * 12 * 8 * 8
		idx := 0
		for c := 0; c < 12; c++ {
			for r := 0; r < 8; r++ {
				for f := 0; f < 8; f++ {
					inputData[offset+idx] = float64(boardTensor[c][r][f])
					idx++
				}
			}
		}

		targetVec, err := ConvertMoveToTarget(entry.FromSquare, entry.ToSquare)
		if err != nil {
			return 0, 0, fmt.Errorf("failed to create target for entry %d: %w", i, err)
		}
		copy(targetData[i*4096:(i+1)*4096], targetVec)
	}

	// Create tensors
	inputTensor := tensor.New(
		tensor.WithShape(batchSize, 12, 8, 8),
		tensor.WithBacking(inputData),
	)

	targetTensor := tensor.New(
		tensor.WithShape(batchSize, 4096),
		tensor.WithBacking(targetData),
	)

	// Set input and target
	if err := gorgonia.Let(t.model.input, inputTensor); err != nil {
		return 0, 0, fmt.Errorf("failed to set input: %w", err)
	}

	if err := gorgonia.Let(t.targetNode, targetTensor); err != nil {
		return 0, 0, fmt.Errorf("failed to set target: %w", err)
	}

	// Run forward and backward pass
	if err := t.model.vm.RunAll(); err != nil {
		return 0, 0, fmt.Errorf("failed to run forward/backward: %w", err)
	}

	// Get loss (only for actual samples)
	lossValue := t.lossNode.Value()
	if lossValue == nil {
		return 0, 0, fmt.Errorf("loss value is nil")
	}

	// Extract scalar loss value
	var avgLoss float64
	switch v := lossValue.Data().(type) {
	case float64:
		avgLoss = v
	case []float64:
		if len(v) > 0 {
			avgLoss = v[0]
		} else {
			return 0, 0, fmt.Errorf("loss value array is empty")
		}
	default:
		return 0, 0, fmt.Errorf("unexpected loss value type: %T", v)
	}

	// Update weights
	learnables := t.model.Learnables()
	valueGrads := make([]gorgonia.ValueGrad, len(learnables))
	for i, n := range learnables {
		valueGrads[i] = n
	}
	if err := t.solver.Step(valueGrads); err != nil {
		return 0, 0, fmt.Errorf("failed to update weights: %w", err)
	}

	// Reset VM
	t.model.vm.Reset()

	// Count correct predictions (only for actual samples)
	correctCount := 0
	outputValue := t.model.output.Value()
	if outputValue != nil {
		outputData := outputValue.Data().([]float64)

		for i := 0; i < actualSize; i++ {
			entry := entries[i]
			maxIdx := 0
			maxProb := outputData[i*4096]
			for j := 1; j < 4096; j++ {
				if outputData[i*4096+j] > maxProb {
					maxProb = outputData[i*4096+j]
					maxIdx = j
				}
			}

			expectedIdx := entry.FromSquare*64 + entry.ToSquare
			if maxIdx == expectedIdx {
				correctCount++
			}
		}
	}

	return avgLoss, correctCount, nil
}

// GetMetrics returns training metrics
func (t *Trainer) GetMetrics() []TrainingMetrics {
	return t.metrics
}

// GetModel returns the underlying CNN model
func (t *Trainer) GetModel() *ChessCNN {
	return t.model
}

// TrainWithCallback trains with a callback function for progress monitoring
func (t *Trainer) TrainWithCallback(dataset *data.Dataset, callback func(TrainingMetrics)) error {
	// Store original verbose setting
	originalVerbose := t.config.Verbose
	t.config.Verbose = false // Disable internal logging

	// Get total samples
	totalSamples, err := dataset.Count()
	if err != nil {
		return fmt.Errorf("failed to get dataset count: %w", err)
	}

	if totalSamples == 0 {
		return fmt.Errorf("dataset is empty")
	}

	// Training loop
	for epoch := 0; epoch < t.config.Epochs; epoch++ {
		startTime := time.Now()

		// Apply learning rate decay
		if epoch > 0 && epoch%t.config.LRDecaySteps == 0 {
			newLR := t.config.LearningRate * math.Pow(t.config.LRDecayRate, float64(epoch/t.config.LRDecaySteps))
			t.solver = gorgonia.NewAdamSolver(
				gorgonia.WithLearnRate(newLR),
				gorgonia.WithBatchSize(float64(t.config.BatchSize)),
				gorgonia.WithClip(t.config.GradientClipMax),
			)
		}

		// Train one epoch
		epochLoss, accuracy, samplesSeen, err := t.trainEpoch(dataset, totalSamples)
		if err != nil {
			return fmt.Errorf("epoch %d failed: %w", epoch+1, err)
		}

		duration := time.Since(startTime)

		// Record metrics
		metrics := TrainingMetrics{
			Epoch:        epoch + 1,
			Loss:         epochLoss,
			Accuracy:     accuracy,
			LearningRate: t.config.LearningRate,
			Duration:     duration,
			SamplesSeen:  samplesSeen,
		}
		t.metrics = append(t.metrics, metrics)

		// Call callback
		if callback != nil {
			callback(metrics)
		}

		// Save checkpoint
		if t.config.SaveInterval > 0 && (epoch+1)%t.config.SaveInterval == 0 {
			if err := t.model.SaveModel(t.config.SavePath); err != nil {
				fmt.Printf("Warning: Failed to save checkpoint: %v\n", err)
			}
		}
	}

	// Final save
	if t.config.SavePath != "" {
		if err := t.model.SaveModel(t.config.SavePath); err != nil {
			return fmt.Errorf("failed to save final model: %w", err)
		}
	}

	// Restore verbose setting
	t.config.Verbose = originalVerbose

	return nil
}
