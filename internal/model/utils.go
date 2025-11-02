package model

import (
	"fmt"
	"math"
)

// ModelInfo contains metadata about the model
type ModelInfo struct {
	InputSize    int
	HiddenSize   int
	OutputSize   int
	TotalParams  int
	ModelVersion string
}

// GetModelInfo returns information about the network
func (cn *ChessNet) GetModelInfo() *ModelInfo {
	totalParams := 0

	learnables := cn.Learnables()
	for _, node := range learnables {
		val := node.Value()
		if val != nil {
			shape := val.Shape()
			params := 1
			for _, dim := range shape {
				params *= dim
			}
			totalParams += params
		}
	}

	return &ModelInfo{
		InputSize:    cn.inputSize,
		HiddenSize:   cn.hiddenSize,
		OutputSize:   cn.outputSize,
		TotalParams:  totalParams,
		ModelVersion: "1.0.0",
	}
}

// ValidateInput checks if the input is valid
func (cn *ChessNet) ValidateInput(input []float64) error {
	if len(input) != cn.inputSize {
		return fmt.Errorf("invalid input size: expected %d, got %d", cn.inputSize, len(input))
	}

	// Check for NaN or Inf
	for i, val := range input {
		if math.IsNaN(val) || math.IsInf(val, 0) {
			return fmt.Errorf("invalid value at index %d: %f", i, val)
		}
	}

	return nil
}

// PredictWithValidation performs inference with input validation
func (cn *ChessNet) PredictWithValidation(boardState []float64) ([]float64, error) {
	if err := cn.ValidateInput(boardState); err != nil {
		return nil, err
	}

	return cn.Predict(boardState)
}

// GetPredictionConfidence returns the confidence of the top prediction
func GetPredictionConfidence(predictions []float64) float64 {
	if len(predictions) == 0 {
		return 0.0
	}

	maxProb := predictions[0]
	for i := 1; i < len(predictions); i++ {
		if predictions[i] > maxProb {
			maxProb = predictions[i]
		}
	}

	return maxProb
}

// GetPredictionEntropy calculates the entropy of predictions
func GetPredictionEntropy(predictions []float64) float64 {
	entropy := 0.0
	for _, p := range predictions {
		if p > 0 {
			entropy -= p * math.Log2(p)
		}
	}
	return entropy
}

// IsHighConfidence checks if prediction confidence is above threshold
func IsHighConfidence(predictions []float64, threshold float64) bool {
	confidence := GetPredictionConfidence(predictions)
	return confidence >= threshold
}

// NormalizePredictions ensures predictions sum to 1.0
func NormalizePredictions(predictions []float64) []float64 {
	sum := 0.0
	for _, p := range predictions {
		sum += p
	}

	if sum == 0 {
		// Uniform distribution if sum is zero
		uniform := 1.0 / float64(len(predictions))
		result := make([]float64, len(predictions))
		for i := range result {
			result[i] = uniform
		}
		return result
	}

	normalized := make([]float64, len(predictions))
	for i, p := range predictions {
		normalized[i] = p / sum
	}

	return normalized
}

// SoftmaxManual applies softmax activation manually (for validation)
func SoftmaxManual(logits []float64) []float64 {
	// Find max for numerical stability
	maxLogit := logits[0]
	for i := 1; i < len(logits); i++ {
		if logits[i] > maxLogit {
			maxLogit = logits[i]
		}
	}

	// Compute exp(x - max)
	expValues := make([]float64, len(logits))
	sum := 0.0
	for i, logit := range logits {
		expValues[i] = math.Exp(logit - maxLogit)
		sum += expValues[i]
	}

	// Normalize
	for i := range expValues {
		expValues[i] /= sum
	}

	return expValues
}

// CompareModels compares two models' predictions
func CompareModels(net1, net2 *ChessNet, testInput []float64) (float64, error) {
	pred1, err := net1.Predict(testInput)
	if err != nil {
		return 0, fmt.Errorf("model 1 prediction failed: %w", err)
	}

	pred2, err := net2.Predict(testInput)
	if err != nil {
		return 0, fmt.Errorf("model 2 prediction failed: %w", err)
	}

	if len(pred1) != len(pred2) {
		return 0, fmt.Errorf("prediction sizes differ: %d vs %d", len(pred1), len(pred2))
	}

	// Calculate mean squared error
	mse := 0.0
	for i := range pred1 {
		diff := pred1[i] - pred2[i]
		mse += diff * diff
	}
	mse /= float64(len(pred1))

	return mse, nil
}

// SimplifiedMoveDecoder converts move index to simple notation
func SimplifiedMoveDecoder(moveIndex int, boardSize int) string {
	row := moveIndex / boardSize
	col := moveIndex % boardSize
	file := rune('a' + col)
	rank := row + 1
	return fmt.Sprintf("%c%d", file, rank)
}

// MoveProbability represents a move with its probability
type MoveProbability struct {
	Square      string
	Index       int
	Probability float64
}

// GetTopMovesDetailed returns detailed move predictions
func GetTopMovesDetailed(predictions []float64, k int) []MoveProbability {
	if k > len(predictions) {
		k = len(predictions)
	}

	moves := GetTopKMoves(predictions, k)

	detailed := make([]MoveProbability, len(moves))
	for i, move := range moves {
		detailed[i] = MoveProbability{
			Square:      SimplifiedMoveDecoder(move.MoveIndex, 8),
			Index:       move.MoveIndex,
			Probability: move.Score,
		}
	}

	return detailed
}

// ModelStatistics contains model statistics
type ModelStatistics struct {
	TotalInferences int64
	AverageLatency  float64
	ErrorCount      int64
}

// StatTracker tracks model usage statistics
type StatTracker struct {
	stats ModelStatistics
}

// NewStatTracker creates a new statistics tracker
func NewStatTracker() *StatTracker {
	return &StatTracker{}
}

// RecordInference records an inference event
func (st *StatTracker) RecordInference(latency float64, success bool) {
	st.stats.TotalInferences++

	// Update running average
	oldAvg := st.stats.AverageLatency
	n := float64(st.stats.TotalInferences)
	st.stats.AverageLatency = (oldAvg*(n-1) + latency) / n

	if !success {
		st.stats.ErrorCount++
	}
}

// GetStatistics returns current statistics
func (st *StatTracker) GetStatistics() ModelStatistics {
	return st.stats
}

// Reset resets the statistics
func (st *StatTracker) Reset() {
	st.stats = ModelStatistics{}
}
