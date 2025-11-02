package training
package training

import (
	"encoding/json"
	"fmt"
	"time"
)

// ReplayEntry represents a single observation-prediction-outcome cycle
type ReplayEntry struct {
	// Board state as tensor
	StateTensor [12][8][8]float32 `json:"state_tensor"`
	
	// Model's prediction
	PredictedMove Move `json:"predicted_move"`
	
	// Actual move played
	ActualMove Move `json:"actual_move"`
	
	// Reward signal (+1 correct, -1 incorrect, 0 partial)
	Reward float64 `json:"reward"`
	
	// Metadata
	Timestamp   int64   `json:"timestamp"`
	GameID      string  `json:"game_id,omitempty"`
	Position    int     `json:"position"` // Move number in game
	Confidence  float64 `json:"confidence,omitempty"`
	
	// Analysis
	IsCorrect   bool    `json:"is_correct"`
	WasInTopK   bool    `json:"was_in_top_k"` // Was actual move in top-K predictions?
	TopKRank    int     `json:"top_k_rank,omitempty"` // Rank of actual move in predictions
}

// Move represents a chess move
type Move struct {
	FromSquare int    `json:"from_square"` // 0-63
	ToSquare   int    `json:"to_square"`   // 0-63
	Notation   string `json:"notation"`    // e.g., "e2e4"
}

// ReplayBuffer manages the collection of replay entries
type ReplayBuffer struct {
	Entries     []ReplayEntry
	MaxSize     int
	TotalAdded  int64
	CorrectPredictions int64
	TotalPredictions   int64
}

// ReplayStats provides statistics about the replay buffer
type ReplayStats struct {
	TotalEntries       int     `json:"total_entries"`
	CorrectPredictions int     `json:"correct_predictions"`
	TotalPredictions   int     `json:"total_predictions"`
	Accuracy           float64 `json:"accuracy"`
	AverageReward      float64 `json:"average_reward"`
	TopKAccuracy       float64 `json:"top_k_accuracy"`
	RecentAccuracy     float64 `json:"recent_accuracy"` // Last 100 predictions
	BufferUtilization  float64 `json:"buffer_utilization"`
}

// NewReplayBuffer creates a new replay buffer
func NewReplayBuffer(maxSize int) *ReplayBuffer {
	return &ReplayBuffer{
		Entries: make([]ReplayEntry, 0, maxSize),
		MaxSize: maxSize,
	}
}

// Add adds a new replay entry to the buffer
func (rb *ReplayBuffer) Add(entry ReplayEntry) {
	// Calculate reward based on correctness
	if entry.PredictedMove.FromSquare == entry.ActualMove.FromSquare &&
		entry.PredictedMove.ToSquare == entry.ActualMove.ToSquare {
		entry.Reward = 1.0
		entry.IsCorrect = true
		rb.CorrectPredictions++
	} else {
		entry.Reward = -1.0
		entry.IsCorrect = false
	}
	
	// Add timestamp if not set
	if entry.Timestamp == 0 {
		entry.Timestamp = time.Now().Unix()
	}
	
	rb.TotalAdded++
	rb.TotalPredictions++
	
	// Add to buffer
	if len(rb.Entries) < rb.MaxSize {
		rb.Entries = append(rb.Entries, entry)
	} else {
		// Replace oldest entry (FIFO)
		copy(rb.Entries, rb.Entries[1:])
		rb.Entries[len(rb.Entries)-1] = entry
	}
}

// GetStats returns current buffer statistics
func (rb *ReplayBuffer) GetStats() ReplayStats {
	if len(rb.Entries) == 0 {
		return ReplayStats{}
	}
	
	stats := ReplayStats{
		TotalEntries:     len(rb.Entries),
		TotalPredictions: int(rb.TotalPredictions),
		BufferUtilization: float64(len(rb.Entries)) / float64(rb.MaxSize),
	}
	
	// Calculate overall accuracy
	if rb.TotalPredictions > 0 {
		stats.Accuracy = float64(rb.CorrectPredictions) / float64(rb.TotalPredictions)
		stats.CorrectPredictions = int(rb.CorrectPredictions)
	}
	
	// Calculate average reward and top-K accuracy
	var totalReward float64
	var topKCorrect int
	
	for _, entry := range rb.Entries {
		totalReward += entry.Reward
		if entry.WasInTopK {
			topKCorrect++
		}
	}
	
	stats.AverageReward = totalReward / float64(len(rb.Entries))
	stats.TopKAccuracy = float64(topKCorrect) / float64(len(rb.Entries))
	
	// Calculate recent accuracy (last 100 entries)
	recentSize := 100
	if len(rb.Entries) < recentSize {
		recentSize = len(rb.Entries)
	}
	
	recentCorrect := 0
	for i := len(rb.Entries) - recentSize; i < len(rb.Entries); i++ {
		if rb.Entries[i].IsCorrect {
			recentCorrect++
		}
	}
	stats.RecentAccuracy = float64(recentCorrect) / float64(recentSize)
	
	return stats
}

// GetRewardWeightedSample returns a sample weighted by rewards
// Entries with positive rewards are more likely to be selected
func (rb *ReplayBuffer) GetRewardWeightedSample(batchSize int) []ReplayEntry {
	if len(rb.Entries) == 0 {
		return nil
	}
	
	if batchSize > len(rb.Entries) {
		batchSize = len(rb.Entries)
	}
	
	// For now, use simple weighted sampling
	// Positive reward entries have 2x probability
	var weighted []ReplayEntry
	
	for _, entry := range rb.Entries {
		weighted = append(weighted, entry)
		if entry.Reward > 0 {
			// Add positive examples twice (higher weight)
			weighted = append(weighted, entry)
		}
	}
	
	// Sample from weighted list
	sample := make([]ReplayEntry, 0, batchSize)
	step := len(weighted) / batchSize
	if step == 0 {
		step = 1
	}
	
	for i := 0; i < len(weighted) && len(sample) < batchSize; i += step {
		sample = append(sample, weighted[i])
	}
	
	return sample
}

// GetBalancedSample returns a balanced sample of correct and incorrect predictions
func (rb *ReplayBuffer) GetBalancedSample(batchSize int) []ReplayEntry {
	if len(rb.Entries) == 0 {
		return nil
	}
	
	var correct, incorrect []ReplayEntry
	for _, entry := range rb.Entries {
		if entry.IsCorrect {
			correct = append(correct, entry)
		} else {
			incorrect = append(incorrect, entry)
		}
	}
	
	// Balance the sample
	halfSize := batchSize / 2
	sample := make([]ReplayEntry, 0, batchSize)
	
	// Add correct examples
	correctStep := len(correct) / halfSize
	if correctStep == 0 {
		correctStep = 1
	}
	for i := 0; i < len(correct) && len(sample) < halfSize; i += correctStep {
		sample = append(sample, correct[i])
	}
	
	// Add incorrect examples
	incorrectStep := len(incorrect) / halfSize
	if incorrectStep == 0 {
		incorrectStep = 1
	}
	for i := 0; i < len(incorrect) && len(sample) < batchSize; i += incorrectStep {
		sample = append(sample, incorrect[i])
	}
	
	return sample
}

// Clear clears the buffer
func (rb *ReplayBuffer) Clear() {
	rb.Entries = make([]ReplayEntry, 0, rb.MaxSize)
}

// ToJSON serializes the buffer to JSON
func (rb *ReplayBuffer) ToJSON() ([]byte, error) {
	return json.MarshalIndent(rb, "", "  ")
}

// FromJSON deserializes the buffer from JSON
func (rb *ReplayBuffer) FromJSON(data []byte) error {
	return json.Unmarshal(data, rb)
}

// String returns a string representation
func (stats ReplayStats) String() string {
	return fmt.Sprintf(
		"ReplayStats{Entries: %d, Accuracy: %.2f%%, Recent: %.2f%%, TopK: %.2f%%, AvgReward: %.2f}",
		stats.TotalEntries,
		stats.Accuracy*100,
		stats.RecentAccuracy*100,
		stats.TopKAccuracy*100,
		stats.AverageReward,
	)
}
