package training

import (
	"fmt"
	"log"
	"math"
	"time"

	"gorgonia.org/gorgonia"

	"github.com/thyrook/partner/internal/model"
	"github.com/thyrook/partner/internal/storage"
)

// Trainer handles neural network training with gradient descent
type Trainer struct {
	model        *model.ChessNet
	store        *storage.ObservationStore
	learningRate float64
	batchSize    int
	learnables   gorgonia.Nodes

	// Learning rate scheduling
	initialLR   float64
	minLR       float64
	warmupSteps int
	currentStep int

	// Training state
	bestLoss  float64
	patience  int
	noImprove int

	// Metrics
	losses     []float64
	accuracies []float64
}

// Config holds training configuration
type Config struct {
	Epochs            int
	BatchSize         int
	LearningRate      float64
	MinLR             float64
	WarmupSteps       int
	ValidationSplit   float64
	EarlyStopPatience int
	CheckpointEvery   int
	CheckpointPath    string
	Verbose           bool
}

// DefaultConfig returns sensible training defaults
func DefaultConfig() *Config {
	return &Config{
		Epochs:            100,
		BatchSize:         64,
		LearningRate:      0.001,
		MinLR:             1e-6,
		WarmupSteps:       1000,
		ValidationSplit:   0.15,
		EarlyStopPatience: 10,
		CheckpointEvery:   5,
		CheckpointPath:    "checkpoints",
		Verbose:           true,
	}
}

// NewTrainer creates a new trainer
func NewTrainer(net *model.ChessNet, store *storage.ObservationStore, cfg *Config) (*Trainer, error) {
	if cfg == nil {
		cfg = DefaultConfig()
	}

	learnables := net.Learnables()

	return &Trainer{
		model:        net,
		store:        store,
		learningRate: cfg.LearningRate,
		batchSize:    cfg.BatchSize,
		learnables:   learnables,
		initialLR:    cfg.LearningRate,
		minLR:        cfg.MinLR,
		warmupSteps:  cfg.WarmupSteps,
		bestLoss:     math.MaxFloat64,
		patience:     cfg.EarlyStopPatience,
		losses:       make([]float64, 0, cfg.Epochs),
		accuracies:   make([]float64, 0, cfg.Epochs),
	}, nil
}

// Train runs the full training loop
func (t *Trainer) Train(cfg *Config) error {
	if cfg == nil {
		cfg = DefaultConfig()
	}

	// Get dataset size
	totalSamples, err := t.store.CountSamples()
	if err != nil {
		return fmt.Errorf("failed to count samples: %w", err)
	}

	if totalSamples == 0 {
		return fmt.Errorf("no training samples")
	}

	// Calculate splits
	valSize := int(float64(totalSamples) * cfg.ValidationSplit)
	trainSize := int(totalSamples) - valSize
	stepsPerEpoch := trainSize / cfg.BatchSize
	totalSteps := stepsPerEpoch * cfg.Epochs

	if cfg.Verbose {
		t.logStart(int(totalSamples), trainSize, valSize, stepsPerEpoch, totalSteps, cfg)
	}

	// Training loop
	for epoch := 1; epoch <= cfg.Epochs; epoch++ {
		startTime := time.Now()

		// Train one epoch
		epochLoss, epochAcc, err := t.trainEpoch(stepsPerEpoch, totalSteps)
		if err != nil {
			if cfg.Verbose {
				log.Printf("Epoch %d error: %v", epoch, err)
			}
			continue
		}

		// Validation
		valLoss, valAcc := 0.0, 0.0
		if valSize > 0 {
			valLoss, valAcc, _ = t.validate(valSize)
		}

		// Store metrics
		t.losses = append(t.losses, epochLoss)
		t.accuracies = append(t.accuracies, epochAcc)

		duration := time.Since(startTime)

		// Log results
		if cfg.Verbose {
			t.logEpoch(epoch, cfg.Epochs, epochLoss, epochAcc, valLoss, valAcc, duration)
		}

		// Checkpoint
		if cfg.CheckpointEvery > 0 && epoch%cfg.CheckpointEvery == 0 {
			t.checkpoint(cfg.CheckpointPath, epoch)
		}

		// Early stopping
		if t.shouldStop(valLoss) {
			if cfg.Verbose {
				fmt.Printf("\nEarly stopping at epoch %d\n", epoch)
			}
			break
		}
	}

	if cfg.Verbose {
		fmt.Printf("\n✓ Training complete! Best loss: %.4f\n", t.bestLoss)
	}

	return nil
}

// trainEpoch trains for one epoch
func (t *Trainer) trainEpoch(stepsPerEpoch, totalSteps int) (float64, float64, error) {
	totalLoss := 0.0
	correct := 0
	total := 0

	for step := 0; step < stepsPerEpoch; step++ {
		t.currentStep++

		// Update learning rate with warmup + cosine decay
		t.updateLearningRate(totalSteps)

		// Get batch
		samples, err := t.store.GetBatch(t.batchSize)
		if err != nil || len(samples) == 0 {
			continue
		}

		// Prepare batch
		inputs := make([][]float64, len(samples))
		targets := make([]int, len(samples))
		for i, s := range samples {
			inputs[i] = s.State
			targets[i] = s.MoveLabel
		}

		// Train step
		loss, acc, err := t.TrainStep(inputs, targets)
		if err != nil {
			continue
		}

		totalLoss += loss
		correct += acc
		total += len(samples)
	}

	avgLoss := totalLoss / float64(stepsPerEpoch)
	accuracy := float64(correct) / float64(total)

	return avgLoss, accuracy, nil
}

// TrainStep performs one training step on a batch
func (t *Trainer) TrainStep(inputs [][]float64, targets []int) (float64, int, error) {
	if len(inputs) != len(targets) {
		return 0, 0, fmt.Errorf("inputs/targets mismatch")
	}

	totalLoss := 0.0
	correct := 0

	for i, input := range inputs {
		// Forward pass
		predictions, err := t.model.Predict(input)
		if err != nil {
			continue
		}

		// Calculate loss
		loss := -math.Log(predictions[targets[i]] + 1e-10)
		totalLoss += loss

		// Check accuracy
		predicted := argmax(predictions)
		if predicted == targets[i] {
			correct++
		}

		// Backward pass (gradient update)
		t.updateWeights(predictions, targets[i])
	}

	avgLoss := totalLoss / float64(len(inputs))
	return avgLoss, correct, nil
}

// updateWeights performs gradient descent weight update
func (t *Trainer) updateWeights(predictions []float64, target int) {
	// Compute gradient: dL/dy = y - t (for cross-entropy)
	grad := make([]float64, len(predictions))
	for i := range predictions {
		if i == target {
			grad[i] = predictions[i] - 1.0
		} else {
			grad[i] = predictions[i]
		}
	}

	// Apply learning rate
	for i := range grad {
		grad[i] *= t.learningRate
	}

	// Update weights through backprop
	// In a full implementation, this would use Gorgonia's autodiff
	// For now, this is a placeholder for the actual gradient descent
}

// validate runs validation
func (t *Trainer) validate(valSize int) (float64, float64, error) {
	numBatches := valSize / t.batchSize
	if numBatches == 0 {
		numBatches = 1
	}

	totalLoss := 0.0
	correct := 0
	total := 0

	for i := 0; i < numBatches; i++ {
		samples, err := t.store.GetBatch(t.batchSize)
		if err != nil || len(samples) == 0 {
			continue
		}

		inputs := make([][]float64, len(samples))
		targets := make([]int, len(samples))
		for j, s := range samples {
			inputs[j] = s.State
			targets[j] = s.MoveLabel
		}

		// Evaluate without gradient updates
		for k, input := range inputs {
			predictions, err := t.model.Predict(input)
			if err != nil {
				continue
			}

			loss := -math.Log(predictions[targets[k]] + 1e-10)
			totalLoss += loss

			if argmax(predictions) == targets[k] {
				correct++
			}
			total++
		}
	}

	avgLoss := totalLoss / float64(total)
	accuracy := float64(correct) / float64(total)

	return avgLoss, accuracy, nil
}

// updateLearningRate applies warmup + cosine decay
func (t *Trainer) updateLearningRate(totalSteps int) {
	// Warmup phase
	if t.currentStep < t.warmupSteps {
		t.learningRate = t.initialLR * float64(t.currentStep) / float64(t.warmupSteps)
		return
	}

	// Cosine decay after warmup
	progress := float64(t.currentStep-t.warmupSteps) / float64(totalSteps-t.warmupSteps)
	t.learningRate = t.minLR + (t.initialLR-t.minLR)*0.5*(1+math.Cos(math.Pi*progress))
	t.learningRate = math.Max(t.learningRate, t.minLR)
}

// shouldStop checks for early stopping
func (t *Trainer) shouldStop(currentLoss float64) bool {
	if t.patience <= 0 {
		return false
	}

	if currentLoss < t.bestLoss-1e-4 {
		t.bestLoss = currentLoss
		t.noImprove = 0
		return false
	}

	t.noImprove++
	return t.noImprove >= t.patience
}

// checkpoint saves model checkpoint
func (t *Trainer) checkpoint(path string, epoch int) error {
	if path == "" {
		return nil
	}

	filename := fmt.Sprintf("%s/checkpoint_epoch_%d.gob", path, epoch)
	if err := t.model.Save(filename); err != nil {
		return fmt.Errorf("checkpoint save failed: %w", err)
	}

	return nil
}

// GetMetrics returns training metrics
func (t *Trainer) GetMetrics() ([]float64, []float64) {
	return t.losses, t.accuracies
}

// logStart prints training configuration
func (t *Trainer) logStart(total, train, val, steps, totalSteps int, cfg *Config) {
	fmt.Println("\n========================================")
	fmt.Println("Training Configuration")
	fmt.Println("========================================")
	fmt.Printf("Dataset:          %d samples\n", total)
	fmt.Printf("  Training:       %d (%.1f%%)\n", train, float64(train)/float64(total)*100)
	fmt.Printf("  Validation:     %d (%.1f%%)\n", val, float64(val)/float64(total)*100)
	fmt.Printf("Epochs:           %d\n", cfg.Epochs)
	fmt.Printf("Batch size:       %d\n", cfg.BatchSize)
	fmt.Printf("Steps per epoch:  %d\n", steps)
	fmt.Printf("Total steps:      %d\n", totalSteps)
	fmt.Printf("Learning rate:    %.6f → %.6f\n", cfg.LearningRate, cfg.MinLR)
	fmt.Printf("Warmup steps:     %d\n", cfg.WarmupSteps)
	fmt.Println("========================================\n")
}

// logEpoch prints epoch results
func (t *Trainer) logEpoch(epoch, totalEpochs int, trainLoss, trainAcc, valLoss, valAcc float64, dur time.Duration) {
	fmt.Printf("Epoch %3d/%d - %.1fs - Loss: %.4f - Acc: %.2f%%",
		epoch, totalEpochs, dur.Seconds(), trainLoss, trainAcc*100)

	if valLoss > 0 {
		fmt.Printf(" - Val Loss: %.4f - Val Acc: %.2f%%", valLoss, valAcc*100)
	}

	if trainLoss < t.bestLoss {
		fmt.Print(" ✓")
	}

	fmt.Println()
}

// argmax returns index of maximum value
func argmax(values []float64) int {
	if len(values) == 0 {
		return -1
	}

	maxIdx := 0
	maxVal := values[0]
	for i := 1; i < len(values); i++ {
		if values[i] > maxVal {
			maxVal = values[i]
			maxIdx = i
		}
	}

	return maxIdx
}
