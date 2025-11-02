package training

import (
	"fmt"
	"math"

	"github.com/thyrook/partner/internal/model"
	"github.com/thyrook/partner/internal/storage"
)

// ImprovedTrainer provides enhanced training with warmup, scheduling, and validation
type ImprovedTrainer struct {
	net          *model.ChessNet
	store        *storage.ObservationStore
	baseLR       float64
	minLR        float64
	batchSize    int
	warmupEpochs int
	currentEpoch int

	// Training state
	bestValLoss float64
	patience    int
	patienceCounter int

	// Metrics
	trainLosses []float64
	valLosses   []float64
}

// ImprovedTrainerConfig holds configuration for the improved trainer
type ImprovedTrainerConfig struct {
	BaseLearningRate float64
	MinLearningRate  float64
	BatchSize        int
	WarmupEpochs     int
	Patience         int  // For early stopping
	ValidationSplit  float64
}

// DefaultImprovedConfig returns sensible defaults
func DefaultImprovedConfig() *ImprovedTrainerConfig {
	return &ImprovedTrainerConfig{
		BaseLearningRate: 0.001,
		MinLearningRate:  0.00001,
		BatchSize:        32,  // Smaller batch for better gradients
		WarmupEpochs:     5,   // Warmup over 5 epochs
		Patience:         10,  // Stop if no improvement for 10 epochs
		ValidationSplit:  0.15, // 15% for validation
	}
}

// NewImprovedTrainer creates a trainer with advanced features
func NewImprovedTrainer(net *model.ChessNet, store *storage.ObservationStore, config *ImprovedTrainerConfig) (*ImprovedTrainer, error) {
	if config == nil {
		config = DefaultImprovedConfig()
	}

	return &ImprovedTrainer{
		net:          net,
		store:        store,
		baseLR:       config.BaseLearningRate,
		minLR:        config.MinLearningRate,
		batchSize:    config.BatchSize,
		warmupEpochs: config.WarmupEpochs,
		patience:     config.Patience,
		bestValLoss:  math.MaxFloat64,
		trainLosses:  make([]float64, 0),
		valLosses:    make([]float64, 0),
	}, nil
}

// GetLearningRate calculates learning rate with warmup and cosine annealing
func (t *ImprovedTrainer) GetLearningRate(epoch, totalEpochs int) float64 {
	// Warmup phase: linear increase from 0 to baseLR
	if epoch < t.warmupEpochs {
		return t.baseLR * float64(epoch) / float64(t.warmupEpochs)
	}

	// Cosine annealing after warmup
	epochAfterWarmup := epoch - t.warmupEpochs
	totalAfterWarmup := totalEpochs - t.warmupEpochs
	if totalAfterWarmup <= 0 {
		return t.baseLR
	}

	// Cosine decay
	cosineDecay := 0.5 * (1 + math.Cos(math.Pi*float64(epochAfterWarmup)/float64(totalAfterWarmup)))
	lr := t.minLR + (t.baseLR-t.minLR)*cosineDecay

	return lr
}

// TrainEpochWithValidation trains one epoch with train/val split
func (t *ImprovedTrainer) TrainEpochWithValidation(epoch, totalEpochs int, valSplit float64) (trainLoss, valLoss float64, shouldStop bool, err error) {
	// Get current learning rate
	lr := t.GetLearningRate(epoch, totalEpochs)

	// Get total samples
	totalSamples, err := t.store.CountSamples()
	if err != nil {
		return 0, 0, false, fmt.Errorf("failed to count samples: %w", err)
	}

	if totalSamples == 0 {
		return 0, 0, false, fmt.Errorf("no samples available")
	}

	// Split into train/val
	valSize := int(float64(totalSamples) * valSplit)
	trainSize := int(totalSamples) - valSize

	if trainSize < t.batchSize {
		return 0, 0, false, fmt.Errorf("not enough training samples: %d < %d", trainSize, t.batchSize)
	}

	// Train on training set
	numTrainBatches := trainSize / t.batchSize
	totalTrainLoss := 0.0

	for i := 0; i < numTrainBatches; i++ {
		// Get batch from training portion
		samples, err := t.store.GetBatch(t.batchSize)
		if err != nil {
			continue
		}

		// Convert to training format
		inputs := make([][]float64, len(samples))
		targets := make([]int, len(samples))

		for j, sample := range samples {
			inputs[j] = sample.State
			targets[j] = sample.MoveLabel
		}

		// Create trainer (with current LR)
		trainer, err := NewTrainer(t.net, lr, t.batchSize)
		if err != nil {
			return 0, 0, false, err
		}

		// Train step
		loss, err := trainer.TrainStep(inputs, targets)
		if err != nil {
			continue
		}

		totalTrainLoss += loss
	}

	trainLoss = totalTrainLoss / float64(numTrainBatches)

	// Validate on validation set
	numValBatches := valSize / t.batchSize
	if numValBatches == 0 {
		numValBatches = 1
	}

	totalValLoss := 0.0
	for i := 0; i < numValBatches; i++ {
		samples, err := t.store.GetBatch(t.batchSize)
		if err != nil {
			continue
		}

		inputs := make([][]float64, len(samples))
		targets := make([]int, len(samples))

		for j, sample := range samples {
			inputs[j] = sample.State
			targets[j] = sample.MoveLabel
		}

		// Evaluate (no training)
		trainer, err := NewTrainer(t.net, 0, t.batchSize)  // LR=0 for eval
		if err != nil {
			continue
		}

		loss, err := trainer.TrainStep(inputs, targets)
		if err != nil {
			continue
		}

		totalValLoss += loss
	}

	valLoss = totalValLoss / float64(numValBatches)

	// Store metrics
	t.trainLosses = append(t.trainLosses, trainLoss)
	t.valLosses = append(t.valLosses, valLoss)

	// Early stopping check
	if valLoss < t.bestValLoss {
		t.bestValLoss = valLoss
		t.patienceCounter = 0
	} else {
		t.patienceCounter++
	}

	shouldStop = t.patienceCounter >= t.patience

	t.currentEpoch = epoch

	return trainLoss, valLoss, shouldStop, nil
}

// GetMetrics returns training history
func (t *ImprovedTrainer) GetMetrics() (trainLosses, valLosses []float64) {
	return t.trainLosses, t.valLosses
}

// TrainWithImprovedSchedule provides complete training loop with all enhancements
func TrainWithImprovedSchedule(net *model.ChessNet, store *storage.ObservationStore, config *ImprovedTrainerConfig, epochs int, verbose bool) error {
	trainer, err := NewImprovedTrainer(net, store, config)
	if err != nil {
		return err
	}

	if verbose {
		fmt.Println("Training Configuration:")
		fmt.Printf("  Base LR:        %.6f\n", config.BaseLearningRate)
		fmt.Printf("  Min LR:         %.6f\n", config.MinLearningRate)
		fmt.Printf("  Batch size:     %d\n", config.BatchSize)
		fmt.Printf("  Warmup epochs:  %d\n", config.WarmupEpochs)
		fmt.Printf("  Patience:       %d\n", config.Patience)
		fmt.Printf("  Val split:      %.1f%%\n", config.ValidationSplit*100)
		fmt.Println()
	}

	for epoch := 1; epoch <= epochs; epoch++ {
		lr := trainer.GetLearningRate(epoch, epochs)
		trainLoss, valLoss, shouldStop, err := trainer.TrainEpochWithValidation(epoch, epochs, config.ValidationSplit)
		
		if err != nil {
			if verbose {
				fmt.Printf("Epoch %d/%d failed: %v\n", epoch, epochs, err)
			}
			continue
		}

		if verbose {
			fmt.Printf("Epoch %d/%d - LR: %.6f - Train Loss: %.4f - Val Loss: %.4f", 
				epoch, epochs, lr, trainLoss, valLoss)
			
			if valLoss < trainer.bestValLoss {
				fmt.Print(" ✓ (best)")
			}
			fmt.Println()
		}

		if shouldStop {
			if verbose {
				fmt.Printf("\nEarly stopping at epoch %d (no improvement for %d epochs)\n", epoch, config.Patience)
			}
			break
		}
	}

	if verbose {
		fmt.Printf("\nTraining complete. Best validation loss: %.4f\n", trainer.bestValLoss)
	}

	return nil
}
