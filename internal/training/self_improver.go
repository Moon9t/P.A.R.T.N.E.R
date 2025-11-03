package training

import (
	"encoding/json"
	"fmt"
	"log"
	"math"
	"time"

	"github.com/thyrook/partner/internal/data"
	"github.com/thyrook/partner/internal/model"
)

// SelfImprover manages the self-improving training loop
type SelfImprover struct {
	model   *model.ChessCNN
	trainer *model.Trainer
	buffer  *ReplayBuffer
	storage *ReplayStorage

	// Configuration
	config ImproverConfig

	// Statistics
	stats ImproverStats

	// State
	lastTrainTime time.Time
	trainingCycle int
}

// ImproverConfig holds configuration for self-improvement
type ImproverConfig struct {
	// Replay buffer settings
	BufferSize         int `json:"buffer_size"`
	MinSamplesForTrain int `json:"min_samples_for_train"`

	// Training settings
	BatchSize        int     `json:"batch_size"`
	LearningRate     float64 `json:"learning_rate"`
	TrainIntervalSec int     `json:"train_interval_sec"`

	// Sampling strategy
	UseRewardWeighting bool `json:"use_reward_weighting"`
	UseBalancedSample  bool `json:"use_balanced_sample"`

	// Evaluation
	EvalBatchSize     int     `json:"eval_batch_size"`
	AccuracyThreshold float64 `json:"accuracy_threshold"`

	// Storage
	DBPath   string `json:"db_path"`
	JSONLDir string `json:"jsonl_dir"`
	AutoSave bool   `json:"auto_save"`
}

// ImproverStats tracks improvement metrics
type ImproverStats struct {
	TotalCycles      int       `json:"total_cycles"`
	TotalSamples     int64     `json:"total_samples"`
	CurrentAccuracy  float64   `json:"current_accuracy"`
	BaselineAccuracy float64   `json:"baseline_accuracy"`
	BestAccuracy     float64   `json:"best_accuracy"`
	ImprovementDelta float64   `json:"improvement_delta"`
	LastTrainTime    time.Time `json:"last_train_time"`
	AvgTrainDuration float64   `json:"avg_train_duration_sec"`

	// Per-cycle history
	AccuracyHistory []float64 `json:"accuracy_history"`
	RewardHistory   []float64 `json:"reward_history"`
}

// DefaultImproverConfig returns default configuration
func DefaultImproverConfig() ImproverConfig {
	return ImproverConfig{
		BufferSize:         10000,
		MinSamplesForTrain: 50,
		BatchSize:          32,
		LearningRate:       0.0001,
		TrainIntervalSec:   300, // 5 minutes
		UseRewardWeighting: true,
		UseBalancedSample:  false,
		EvalBatchSize:      100,
		AccuracyThreshold:  0.6,
		DBPath:             "data/replays/replay.db",
		JSONLDir:           "data/replays",
		AutoSave:           true,
	}
}

// NewSelfImprover creates a new self-improver
func NewSelfImprover(cnn *model.ChessCNN, config ImproverConfig) (*SelfImprover, error) {
	// Create replay buffer
	buffer := NewReplayBuffer(config.BufferSize)

	// Create storage
	storage, err := NewReplayStorage(config.DBPath, config.JSONLDir)
	if err != nil {
		return nil, fmt.Errorf("failed to create storage: %w", err)
	}

	// Load existing entries
	existingEntries, err := storage.LoadAll()
	if err != nil {
		log.Printf("Warning: failed to load existing entries: %v", err)
	} else {
		log.Printf("Loaded %d existing replay entries", len(existingEntries))
		for _, entry := range existingEntries {
			buffer.Add(entry)
		}
	}

	// Create trainer for incremental training
	trainerConfig := &model.TrainingConfig{
		Epochs:          1,
		BatchSize:       config.BatchSize,
		LearningRate:    config.LearningRate,
		LRDecayRate:     1.0,
		LRDecaySteps:    1,
		GradientClipMax: 5.0,
		Verbose:         false,
	}

	trainer, err := model.NewTrainer(trainerConfig)
	if err != nil {
		storage.Close()
		return nil, fmt.Errorf("failed to create trainer: %w", err)
	}

	improver := &SelfImprover{
		model:         cnn,
		trainer:       trainer,
		buffer:        buffer,
		storage:       storage,
		config:        config,
		lastTrainTime: time.Now(),
	}

	// Evaluate baseline accuracy
	if len(buffer.Entries) > 0 {
		improver.stats.BaselineAccuracy = improver.EvaluateAccuracy()
		improver.stats.CurrentAccuracy = improver.stats.BaselineAccuracy
		improver.stats.BestAccuracy = improver.stats.BaselineAccuracy
	}

	return improver, nil
}

// ObservePrediction logs a prediction-outcome pair
func (si *SelfImprover) ObservePrediction(
	stateTensor [12][8][8]float32,
	predicted Move,
	actual Move,
	topKPredictions []Move,
	confidence float64,
) {
	entry := ReplayEntry{
		StateTensor:   stateTensor,
		PredictedMove: predicted,
		ActualMove:    actual,
		Timestamp:     time.Now().Unix(),
		Confidence:    confidence,
	}

	// Check if actual move was in top-K
	entry.WasInTopK = false
	for i, pred := range topKPredictions {
		if pred.FromSquare == actual.FromSquare && pred.ToSquare == actual.ToSquare {
			entry.WasInTopK = true
			entry.TopKRank = i + 1
			break
		}
	}

	// Add to buffer
	si.buffer.Add(entry)
	si.stats.TotalSamples++

	// Auto-save to storage if enabled
	if si.config.AutoSave {
		if err := si.storage.Store(entry); err != nil {
			log.Printf("Warning: failed to store replay entry: %v", err)
		}
	}

	// Check if it's time to train
	si.CheckAndTrain()
}

// CheckAndTrain checks if training should occur and executes if needed
func (si *SelfImprover) CheckAndTrain() bool {
	// Check if enough samples
	if len(si.buffer.Entries) < si.config.MinSamplesForTrain {
		return false
	}

	// Check if enough time has passed
	timeSinceLastTrain := time.Since(si.lastTrainTime)
	if timeSinceLastTrain.Seconds() < float64(si.config.TrainIntervalSec) {
		return false
	}

	// Execute training
	if err := si.Train(); err != nil {
		log.Printf("Training failed: %v", err)
		return false
	}

	return true
}

// Train executes a training cycle on the replay buffer
func (si *SelfImprover) Train() error {
	log.Printf("Starting self-improvement training cycle %d", si.trainingCycle+1)
	startTime := time.Now()

	// Get sample from buffer
	var sample []ReplayEntry
	if si.config.UseRewardWeighting {
		sample = si.buffer.GetRewardWeightedSample(si.config.BatchSize)
	} else if si.config.UseBalancedSample {
		sample = si.buffer.GetBalancedSample(si.config.BatchSize)
	} else {
		// Use recent entries
		if len(si.buffer.Entries) <= si.config.BatchSize {
			sample = si.buffer.Entries
		} else {
			start := len(si.buffer.Entries) - si.config.BatchSize
			sample = si.buffer.Entries[start:]
		}
	}

	if len(sample) == 0 {
		return fmt.Errorf("no samples available for training")
	}

	log.Printf("Training on %d samples (reward-weighted: %v)",
		len(sample), si.config.UseRewardWeighting)

	// Prepare training data - convert ReplayEntry to DataEntry format
	log.Printf("Training model on %d samples (correct: %d, incorrect: %d)",
		len(sample), countCorrect(sample), len(sample)-countCorrect(sample))

	entries := make([]*data.DataEntry, len(sample))
	for i, entry := range sample {
		flatTensor := data.TensorToFlatArray(entry.StateTensor)
		entries[i] = &data.DataEntry{
			StateTensor: flatTensor,
			FromSquare:  entry.ActualMove.Index / 64,
			ToSquare:    entry.ActualMove.Index % 64,
		}
	}

	// Train on this batch using the persistent trainer
	loss, correct, err := si.trainer.TrainOnBatch(entries)
	if err != nil {
		return fmt.Errorf("training failed: %w", err)
	}
	
	batchAccuracy := float64(correct) / float64(len(entries)) * 100
	log.Printf("Training complete: loss=%.4f, batch_accuracy=%.2f%% (%d/%d correct)",
		loss, batchAccuracy, correct, len(entries))

	// Evaluate accuracy after training
	oldAccuracy := si.stats.CurrentAccuracy
	newAccuracy := si.EvaluateAccuracy()

	// Update statistics
	si.stats.TotalCycles++
	si.stats.CurrentAccuracy = newAccuracy
	si.stats.ImprovementDelta = newAccuracy - oldAccuracy
	si.stats.LastTrainTime = time.Now()
	si.lastTrainTime = time.Now()
	si.trainingCycle++

	if newAccuracy > si.stats.BestAccuracy {
		si.stats.BestAccuracy = newAccuracy
	}

	// Update history
	si.stats.AccuracyHistory = append(si.stats.AccuracyHistory, newAccuracy)

	bufferStats := si.buffer.GetStats()
	si.stats.RewardHistory = append(si.stats.RewardHistory, bufferStats.AverageReward)

	// Update average train duration
	duration := time.Since(startTime).Seconds()
	if si.stats.AvgTrainDuration == 0 {
		si.stats.AvgTrainDuration = duration
	} else {
		si.stats.AvgTrainDuration = (si.stats.AvgTrainDuration + duration) / 2
	}

	log.Printf("Training cycle %d complete: accuracy %.2f%% -> %.2f%% (Δ%+.2f%%), duration: %.2fs",
		si.trainingCycle,
		oldAccuracy*100,
		newAccuracy*100,
		si.stats.ImprovementDelta*100,
		duration)

	return nil
}

// EvaluateAccuracy evaluates model accuracy on replay buffer
func (si *SelfImprover) EvaluateAccuracy() float64 {
	if len(si.buffer.Entries) == 0 {
		return 0.0
	}

	// Use buffer stats for accuracy
	stats := si.buffer.GetStats()

	// Can also evaluate on a held-out set if available
	// For now, use recent accuracy from buffer
	return stats.RecentAccuracy
}

// GetStats returns current statistics
func (si *SelfImprover) GetStats() ImproverStats {
	return si.stats
}

// GetBufferStats returns replay buffer statistics
func (si *SelfImprover) GetBufferStats() ReplayStats {
	return si.buffer.GetStats()
}

// ExportMetrics exports metrics to a file for visualization
func (si *SelfImprover) ExportMetrics(filename string) error {
	// Create performance data structure
	type PerformanceData struct {
		Stats       ImproverStats  `json:"stats"`
		BufferStats ReplayStats    `json:"buffer_stats"`
		Config      ImproverConfig `json:"config"`
		Timestamp   time.Time      `json:"timestamp"`
		GraphData   GraphData      `json:"graph_data"`
	}

	graphData := GraphData{
		Cycles:   make([]int, len(si.stats.AccuracyHistory)),
		Accuracy: si.stats.AccuracyHistory,
		Rewards:  si.stats.RewardHistory,
	}

	for i := range graphData.Cycles {
		graphData.Cycles[i] = i + 1
	}

	data := PerformanceData{
		Stats:       si.stats,
		BufferStats: si.buffer.GetStats(),
		Config:      si.config,
		Timestamp:   time.Now(),
		GraphData:   graphData,
	}

	// Save to storage metadata
	jsonData, err := json.Marshal(data)
	if err != nil {
		return err
	}

	return si.storage.SetMetadata(filename, string(jsonData))
}

// Close closes the improver and saves state
func (si *SelfImprover) Close() error {
	// Export final metrics
	if err := si.ExportMetrics("final_metrics"); err != nil {
		log.Printf("Warning: failed to export metrics: %v", err)
	}

	// Close trainer's model
	if si.trainer != nil && si.trainer.GetModel() != nil {
		si.trainer.GetModel().Close()
	}

	return si.storage.Close()
}

// GraphData holds data for performance graphs
type GraphData struct {
	Cycles   []int     `json:"cycles"`
	Accuracy []float64 `json:"accuracy"`
	Rewards  []float64 `json:"rewards"`
}

// Helper functions

func countCorrect(entries []ReplayEntry) int {
	count := 0
	for _, entry := range entries {
		if entry.IsCorrect {
			count++
		}
	}
	return count
}

// CalculateImprovement calculates improvement metrics
func (si *SelfImprover) CalculateImprovement() ImprovementMetrics {
	metrics := ImprovementMetrics{
		TotalCycles:      si.stats.TotalCycles,
		BaselineAccuracy: si.stats.BaselineAccuracy,
		CurrentAccuracy:  si.stats.CurrentAccuracy,
		BestAccuracy:     si.stats.BestAccuracy,
	}

	if si.stats.BaselineAccuracy > 0 {
		metrics.RelativeImprovement = (si.stats.CurrentAccuracy - si.stats.BaselineAccuracy) / si.stats.BaselineAccuracy
		metrics.AbsoluteImprovement = si.stats.CurrentAccuracy - si.stats.BaselineAccuracy
	}

	// Calculate trend
	if len(si.stats.AccuracyHistory) >= 2 {
		recent := si.stats.AccuracyHistory[len(si.stats.AccuracyHistory)-1]
		previous := si.stats.AccuracyHistory[len(si.stats.AccuracyHistory)-2]
		metrics.RecentTrend = recent - previous
	}

	// Calculate variance
	if len(si.stats.AccuracyHistory) > 0 {
		mean := si.stats.CurrentAccuracy
		var sumSquares float64
		for _, acc := range si.stats.AccuracyHistory {
			diff := acc - mean
			sumSquares += diff * diff
		}
		metrics.Variance = sumSquares / float64(len(si.stats.AccuracyHistory))
		metrics.StdDev = math.Sqrt(metrics.Variance)
	}

	metrics.IsImproving = metrics.RecentTrend > 0

	return metrics
}

// ImprovementMetrics contains detailed improvement statistics
type ImprovementMetrics struct {
	TotalCycles         int     `json:"total_cycles"`
	BaselineAccuracy    float64 `json:"baseline_accuracy"`
	CurrentAccuracy     float64 `json:"current_accuracy"`
	BestAccuracy        float64 `json:"best_accuracy"`
	RelativeImprovement float64 `json:"relative_improvement"`
	AbsoluteImprovement float64 `json:"absolute_improvement"`
	RecentTrend         float64 `json:"recent_trend"`
	Variance            float64 `json:"variance"`
	StdDev              float64 `json:"std_dev"`
	IsImproving         bool    `json:"is_improving"`
}

// String returns a string representation
func (im ImprovementMetrics) String() string {
	trend := "→"
	if im.IsImproving {
		trend = "↑"
	} else if im.RecentTrend < 0 {
		trend = "↓"
	}

	return fmt.Sprintf(
		"Improvement{Baseline: %.2f%%, Current: %.2f%%, Best: %.2f%%, Δ: %+.2f%% %s}",
		im.BaselineAccuracy*100,
		im.CurrentAccuracy*100,
		im.BestAccuracy*100,
		im.AbsoluteImprovement*100,
		trend,
	)
}
