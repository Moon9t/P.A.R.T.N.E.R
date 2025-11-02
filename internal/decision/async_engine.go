package decision

import (
	"fmt"
	"sync"
	"time"

	"go.uber.org/zap"

	"github.com/thyrook/partner/internal/model"
	"github.com/thyrook/partner/internal/vision"
)

// AsyncDecisionEngine handles decision-making with async capture support
type AsyncDecisionEngine struct {
	model               *model.ChessNet
	asyncCapturer       *vision.AsyncCapturer
	confidenceThreshold float64
	topK                int
	logger              *zap.Logger
	mu                  sync.RWMutex

	// Statistics
	totalDecisions     int
	successfulCaptures int
	failedCaptures     int
	totalInferenceTime time.Duration
	totalDecisionTime  time.Duration
}

// NewAsyncDecisionEngine creates a decision engine with async capture
func NewAsyncDecisionEngine(
	net *model.ChessNet,
	asyncCapturer *vision.AsyncCapturer,
	confidenceThreshold float64,
	topK int,
	logger *zap.Logger,
) *AsyncDecisionEngine {
	return &AsyncDecisionEngine{
		model:               net,
		asyncCapturer:       asyncCapturer,
		confidenceThreshold: confidenceThreshold,
		topK:                topK,
		logger:              logger,
	}
}

// MakeDecision performs decision-making using async-captured frame
// This is much faster than sync version because capture is already done
func (ade *AsyncDecisionEngine) MakeDecision() (*Decision, error) {
	decisionStart := time.Now()

	ade.mu.Lock()
	ade.totalDecisions++
	ade.mu.Unlock()

	// Extract board state from async capture (fast, <1ms)
	boardState, err := ade.asyncCapturer.ExtractBoardState()
	if err != nil {
		ade.mu.Lock()
		ade.failedCaptures++
		ade.mu.Unlock()

		ade.logger.Warn("Failed to get board state from async capture",
			zap.Error(err))
		return nil, fmt.Errorf("board state extraction failed: %w", err)
	}

	ade.mu.Lock()
	ade.successfulCaptures++
	ade.mu.Unlock()

	// Run inference with timing
	inferStart := time.Now()
	predictions, err := ade.model.Predict(boardState.Grid)
	inferDuration := time.Since(inferStart)

	ade.mu.Lock()
	ade.totalInferenceTime += inferDuration
	ade.mu.Unlock()

	if err != nil {
		ade.logger.Error("Inference failed", zap.Error(err))
		return nil, fmt.Errorf("inference failed: %w", err)
	}

	// Rank moves
	rankedMoves := ade.rankMoves(predictions)

	if len(rankedMoves) == 0 {
		return nil, fmt.Errorf("no valid moves found")
	}

	// Build decision
	decision := &Decision{
		TopMove:      &rankedMoves[0],
		Alternatives: rankedMoves[1:],
		Timestamp:    time.Now(),
		InferenceMs:  float64(inferDuration.Microseconds()) / 1000.0,
		BoardState:   boardState.Grid,
		TotalMoves:   len(predictions),
	}

	decisionDuration := time.Since(decisionStart)
	ade.mu.Lock()
	ade.totalDecisionTime += decisionDuration
	ade.mu.Unlock()

	return decision, nil
}

// rankMoves converts predictions to ranked moves
func (ade *AsyncDecisionEngine) rankMoves(predictions []float64) []RankedMove {
	if len(predictions) == 0 {
		return nil
	}

	// Get top K
	topK := ade.topK
	if topK > len(predictions) {
		topK = len(predictions)
	}

	// Find top K predictions
	type scoredMove struct {
		index int
		score float64
	}

	scored := make([]scoredMove, len(predictions))
	for i, score := range predictions {
		scored[i] = scoredMove{index: i, score: score}
	}

	// Simple selection sort for top K
	for i := 0; i < topK && i < len(scored); i++ {
		maxIdx := i
		for j := i + 1; j < len(scored); j++ {
			if scored[j].score > scored[maxIdx].score {
				maxIdx = j
			}
		}
		scored[i], scored[maxIdx] = scored[maxIdx], scored[i]
	}

	// Convert to RankedMove
	ranked := make([]RankedMove, topK)
	for i := 0; i < topK; i++ {
		move := scored[i]
		ranked[i] = RankedMove{
			Move:        decodeMove(move.index),
			Confidence:  move.score,
			Rank:        i + 1,
			MoveIndex:   move.index,
			Explanation: ade.explainMove(move.score, i+1),
			Category:    categorizeMove(move.score),
		}
	}

	return ranked
}

// explainMove generates human-readable explanation
func (ade *AsyncDecisionEngine) explainMove(confidence float64, rank int) string {
	if confidence >= 0.80 {
		return fmt.Sprintf("Highly recommended (rank %d)", rank)
	} else if confidence >= 0.60 {
		return fmt.Sprintf("Solid move, recommended (rank %d)", rank)
	} else if confidence >= 0.40 {
		return fmt.Sprintf("Reasonable alternative (rank %d)", rank)
	} else if confidence >= 0.20 {
		return fmt.Sprintf("Risky, consider carefully (rank %d)", rank)
	}
	return fmt.Sprintf("Low confidence, not recommended (rank %d)", rank)
}

// categorizeMove assigns a category based on confidence
func categorizeMove(confidence float64) string {
	if confidence >= 0.80 {
		return "Excellent"
	} else if confidence >= 0.60 {
		return "Good"
	} else if confidence >= 0.40 {
		return "Fair"
	} else if confidence >= 0.20 {
		return "Risky"
	}
	return "Uncertain"
}

// decodeMove converts move index to chess notation
func decodeMove(moveIndex int) string {
	from := moveIndex / 64
	to := moveIndex % 64

	fromFile := rune('a' + (from % 8))
	fromRank := (from / 8) + 1
	toFile := rune('a' + (to % 8))
	toRank := (to / 8) + 1

	return fmt.Sprintf("%c%d%c%d", fromFile, fromRank, toFile, toRank)
}

// GetStatistics returns engine performance statistics
func (ade *AsyncDecisionEngine) GetStatistics() EngineStats {
	ade.mu.RLock()
	defer ade.mu.RUnlock()

	var avgInferenceMs float64

	if ade.totalDecisions > 0 {
		avgInferenceMs = float64(ade.totalInferenceTime.Microseconds()) / float64(ade.totalDecisions) / 1000.0
	}

	captureSuccessRate := 0.0
	if ade.successfulCaptures+ade.failedCaptures > 0 {
		captureSuccessRate = float64(ade.successfulCaptures) / float64(ade.successfulCaptures+ade.failedCaptures)
	}

	return EngineStats{
		TotalDecisions:     ade.totalDecisions,
		SuccessfulCaptures: ade.successfulCaptures,
		FailedCaptures:     ade.failedCaptures,
		CaptureSuccessRate: captureSuccessRate,
		AvgInferenceMs:     avgInferenceMs,
		TotalInferenceTime: ade.totalInferenceTime,
	}
}

// GetAsyncCaptureStats returns async capture statistics
func (ade *AsyncDecisionEngine) GetAsyncCaptureStats() vision.AsyncCaptureStats {
	return ade.asyncCapturer.GetStatistics()
}
