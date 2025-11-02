package training

import (
	"fmt"

	"github.com/thyrook/partner/internal/model"
	"github.com/thyrook/partner/internal/storage"
)

// StorageTrainer wraps Trainer with storage-backed data loading
type StorageTrainer struct {
	trainer *Trainer
	store   *storage.ObservationStore
}

// NewStorageTrainer creates a trainer that loads data from storage
func NewStorageTrainer(net *model.ChessNet, store *storage.ObservationStore, learningRate float64, batchSize int) (*StorageTrainer, error) {
	trainer, err := NewTrainer(net, learningRate, batchSize)
	if err != nil {
		return nil, err
	}

	return &StorageTrainer{
		trainer: trainer,
		store:   store,
	}, nil
}

// TrainEpoch trains for one epoch using data from storage
func (st *StorageTrainer) TrainEpoch() (float64, error) {
	// Check if we have enough samples
	count, err := st.store.CountSamples()
	if err != nil {
		return 0, fmt.Errorf("failed to count samples: %w", err)
	}

	if count == 0 {
		return 0, fmt.Errorf("no samples available for training")
	}

	// Calculate number of batches
	actualSize, err := st.store.GetActualSize()
	if err != nil {
		return 0, err
	}

	numBatches := actualSize / st.trainer.batchSize
	if numBatches == 0 {
		numBatches = 1
	}

	totalLoss := 0.0

	// Train on multiple batches
	for i := 0; i < numBatches; i++ {
		// Get a batch from storage
		samples, err := st.store.GetBatch(st.trainer.batchSize)
		if err != nil {
			continue // Skip this batch
		}

		// Convert to training format
		inputs := make([][]float64, len(samples))
		targets := make([]int, len(samples))

		for j, sample := range samples {
			inputs[j] = sample.State
			targets[j] = sample.MoveLabel
		}

		// Train on this batch
		loss, err := st.trainer.TrainStep(inputs, targets)
		if err != nil {
			continue
		}

		totalLoss += loss
	}

	avgLoss := totalLoss / float64(numBatches)
	return avgLoss, nil
}

// TrainFromStorage is a convenience function for training with storage
func TrainFromStorage(net *model.ChessNet, store *storage.ObservationStore, config *TrainingConfig) error {
	if config == nil {
		config = DefaultTrainingConfig()
	}

	storageTrainer, err := NewStorageTrainer(net, store, config.LearningRate, config.BatchSize)
	if err != nil {
		return fmt.Errorf("failed to create storage trainer: %w", err)
	}

	// Check minimum samples
	count, err := store.CountSamples()
	if err != nil {
		return err
	}

	if count < uint64(config.BatchSize) {
		return fmt.Errorf("not enough samples: have %d, need at least %d", count, config.BatchSize)
	}

	// Training loop
	for epoch := 0; epoch < config.Epochs; epoch++ {
		loss, err := storageTrainer.TrainEpoch()
		if err != nil {
			if config.Verbose {
				fmt.Printf("Epoch %d failed: %v\n", epoch+1, err)
			}
			continue
		}

		if config.Verbose {
			fmt.Printf("Epoch %d/%d - Loss: %.4f\n", epoch+1, config.Epochs, loss)
		}
	}

	return nil
}

// GetTrainingStats returns statistics about training data
func GetTrainingStats(store *storage.ObservationStore) (map[string]interface{}, error) {
	stats, err := store.GetStats()
	if err != nil {
		return nil, err
	}

	return map[string]interface{}{
		"total_samples":  stats.TotalSamples,
		"actual_samples": stats.ActualSamples,
		"max_size":       stats.MaxSize,
		"is_wrapped":     stats.IsWrapped,
		"utilization":    float64(stats.ActualSamples) / float64(stats.MaxSize) * 100,
	}, nil
}
