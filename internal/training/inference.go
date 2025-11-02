package training

import (
	"fmt"
	"time"

	"github.com/thyrook/partner/internal/model"
	"github.com/thyrook/partner/internal/vision"
)

// InferenceEngine handles real-time move predictions
type InferenceEngine struct {
	net              *model.ChessNet
	confidenceThresh float64
	topK             int
}

// NewInferenceEngine creates a new inference engine
func NewInferenceEngine(net *model.ChessNet, confidenceThresh float64, topK int) *InferenceEngine {
	if topK <= 0 {
		topK = 5
	}

	if confidenceThresh <= 0 {
		confidenceThresh = 0.1
	}

	return &InferenceEngine{
		net:              net,
		confidenceThresh: confidenceThresh,
		topK:             topK,
	}
}

// InferMove predicts the best move from a board state
func (ie *InferenceEngine) InferMove(stateTensor []float64) (*MoveRecommendation, error) {
	if len(stateTensor) != 64 {
		return nil, fmt.Errorf("invalid state tensor size: expected 64, got %d", len(stateTensor))
	}

	startTime := time.Now()

	// Run inference
	predictions, err := ie.net.Predict(stateTensor)
	if err != nil {
		return nil, fmt.Errorf("inference failed: %w", err)
	}

	inferenceTime := time.Since(startTime)

	// Get top K moves
	topMoves := model.GetTopKMoves(predictions, ie.topK)

	// Create recommendation
	rec := &MoveRecommendation{
		PrimaryMove:      convertToMove(topMoves[0]),
		AlternativeMoves: make([]Move, 0, len(topMoves)-1),
		InferenceTime:    inferenceTime,
		BoardState:       stateTensor,
	}

	// Add alternative moves
	for i := 1; i < len(topMoves); i++ {
		if topMoves[i].Score >= ie.confidenceThresh {
			rec.AlternativeMoves = append(rec.AlternativeMoves, convertToMove(topMoves[i]))
		}
	}

	// Determine confidence level
	rec.ConfidenceLevel = categorizeConfidence(topMoves[0].Score)

	return rec, nil
}

// InferFromCapture captures the board and infers a move
func (ie *InferenceEngine) InferFromCapture(capturer *vision.Capturer) (*MoveRecommendation, error) {
	// Capture current board state
	state, err := capturer.ExtractBoardState()
	if err != nil {
		return nil, fmt.Errorf("capture failed: %w", err)
	}

	// Infer move
	rec, err := ie.InferMove(state.Grid)
	if err != nil {
		return nil, err
	}

	// Add capture metadata
	rec.CaptureTime = state.Timestamp
	rec.BoardChanged = state.Changed
	rec.DiffScore = state.DiffScore

	return rec, nil
}

// InferBatch runs inference on multiple board states
func (ie *InferenceEngine) InferBatch(states [][]float64) ([]*MoveRecommendation, error) {
	recommendations := make([]*MoveRecommendation, 0, len(states))

	for i, state := range states {
		rec, err := ie.InferMove(state)
		if err != nil {
			// Log error but continue with other states
			fmt.Printf("Warning: Inference failed for state %d: %v\n", i, err)
			continue
		}
		recommendations = append(recommendations, rec)
	}

	return recommendations, nil
}

// MoveRecommendation contains a move prediction with metadata
type MoveRecommendation struct {
	PrimaryMove      Move
	AlternativeMoves []Move
	ConfidenceLevel  string
	InferenceTime    time.Duration
	CaptureTime      time.Time
	BoardChanged     bool
	DiffScore        float64
	BoardState       []float64
}

// Move represents a chess move with confidence
type Move struct {
	Index      int
	Notation   string
	FromSquare string
	ToSquare   string
	Confidence float64
}

// String returns a formatted string representation
func (m Move) String() string {
	return fmt.Sprintf("%s (%.1f%%)", m.Notation, m.Confidence*100)
}

// convertToMove converts a MoveScore to a Move
func convertToMove(ms model.MoveScore) Move {
	notation := model.DecodeMove(ms.MoveIndex)

	// Extract from and to squares
	fromSquare := ""
	toSquare := ""
	if len(notation) == 4 {
		fromSquare = notation[0:2]
		toSquare = notation[2:4]
	}

	return Move{
		Index:      ms.MoveIndex,
		Notation:   notation,
		FromSquare: fromSquare,
		ToSquare:   toSquare,
		Confidence: ms.Score,
	}
}

// categorizeConfidence categorizes confidence into levels
func categorizeConfidence(confidence float64) string {
	switch {
	case confidence >= 0.8:
		return "Very High"
	case confidence >= 0.6:
		return "High"
	case confidence >= 0.4:
		return "Medium"
	case confidence >= 0.2:
		return "Low"
	default:
		return "Very Low"
	}
}

// FormatRecommendation formats a recommendation for display
func FormatRecommendation(rec *MoveRecommendation) string {
	output := "\n"
	output += "═══════════════════════════════════════════════════════════\n"
	output += "  🎯 MOVE RECOMMENDATION\n"
	output += "═══════════════════════════════════════════════════════════\n"
	output += fmt.Sprintf("\n  Primary Move:     %s → %s\n", rec.PrimaryMove.FromSquare, rec.PrimaryMove.ToSquare)
	output += fmt.Sprintf("  Confidence:       %.1f%% (%s)\n", rec.PrimaryMove.Confidence*100, rec.ConfidenceLevel)
	output += fmt.Sprintf("  Inference Time:   %v\n", rec.InferenceTime.Round(time.Microsecond))

	if len(rec.AlternativeMoves) > 0 {
		output += "\n  Alternative Moves:\n"
		for i, move := range rec.AlternativeMoves {
			if i >= 3 { // Limit to top 3 alternatives
				break
			}
			output += fmt.Sprintf("    %d. %s → %s (%.1f%%)\n",
				i+1, move.FromSquare, move.ToSquare, move.Confidence*100)
		}
	}

	if !rec.CaptureTime.IsZero() {
		output += fmt.Sprintf("\n  Capture Time:     %s\n", rec.CaptureTime.Format("15:04:05.000"))
		output += fmt.Sprintf("  Board Changed:    %v\n", rec.BoardChanged)
		if rec.BoardChanged {
			output += fmt.Sprintf("  Change Score:     %.2f\n", rec.DiffScore)
		}
	}

	output += "\n═══════════════════════════════════════════════════════════\n"

	return output
}

// CompactFormatRecommendation formats a recommendation compactly
func CompactFormatRecommendation(rec *MoveRecommendation) string {
	return fmt.Sprintf("➤ %s (%.1f%%) | Alternatives: %d | Time: %v",
		rec.PrimaryMove.Notation,
		rec.PrimaryMove.Confidence*100,
		len(rec.AlternativeMoves),
		rec.InferenceTime.Round(time.Millisecond))
}

// RecommendationQuality assesses the quality of a recommendation
func RecommendationQuality(rec *MoveRecommendation) string {
	if rec.PrimaryMove.Confidence >= 0.7 && len(rec.AlternativeMoves) >= 2 {
		return "Excellent - High confidence with good alternatives"
	} else if rec.PrimaryMove.Confidence >= 0.5 {
		return "Good - Reasonable confidence"
	} else if rec.PrimaryMove.Confidence >= 0.3 {
		return "Fair - Low confidence, use caution"
	} else {
		return "Poor - Very low confidence, not recommended"
	}
}

// ValidateRecommendation checks if a recommendation meets quality standards
func ValidateRecommendation(rec *MoveRecommendation, minConfidence float64) bool {
	if rec == nil {
		return false
	}

	if rec.PrimaryMove.Confidence < minConfidence {
		return false
	}

	if rec.PrimaryMove.Notation == "" {
		return false
	}

	return true
}
