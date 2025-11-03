package decision

import (
	"fmt"
	"sync"
	"time"

	"go.uber.org/zap"

	"github.com/thyrook/partner/internal/model"
	"github.com/thyrook/partner/internal/vision"
)

// RankedMove represents a move with ranking information
type RankedMove struct {
	Move        string
	Confidence  float64
	Rank        int
	MoveIndex   int
	Explanation string
	Category    string // "Excellent", "Good", "Fair", "Risky"
}

// DecisionEngine handles advanced move ranking and selection
type DecisionEngine struct {
	model               *model.ChessNet
	capturer            *vision.Capturer
	confidenceThreshold float64
	topK                int
	logger              *zap.Logger
	mu                  sync.RWMutex

	// Statistics
	totalDecisions     int
	successfulCaptures int
	failedCaptures     int
	totalInferenceTime time.Duration
}

// NewDecisionEngine creates an enhanced decision engine
func NewDecisionEngine(
	net *model.ChessNet,
	capturer *vision.Capturer,
	confidenceThreshold float64,
	topK int,
	logger *zap.Logger,
) *DecisionEngine {
	return &DecisionEngine{
		model:               net,
		capturer:            capturer,
		confidenceThreshold: confidenceThreshold,
		topK:                topK,
		logger:              logger,
	}
}

// Decision represents a complete decision with ranked alternatives
type Decision struct {
	TopMove      *RankedMove
	Alternatives []RankedMove
	Timestamp    time.Time
	InferenceMs  float64
	BoardState   []float64
	TotalMoves   int
}

// MakeDecision performs complete decision-making process
func (de *DecisionEngine) MakeDecision() (*Decision, error) {
	startTime := time.Now()

	de.mu.Lock()
	de.totalDecisions++
	de.mu.Unlock()

	// Capture board state with fallback
	boardState, err := de.captureBoardWithFallback()
	if err != nil {
		de.mu.Lock()
		de.failedCaptures++
		de.mu.Unlock()

		de.logger.Error("Capture failed",
			zap.Error(err),
			zap.Int("total_failures", de.failedCaptures),
		)
		return nil, fmt.Errorf("capture failed after retries: %w", err)
	}

	de.mu.Lock()
	de.successfulCaptures++
	de.mu.Unlock()

	// Run inference with timing
	inferStart := time.Now()
	predictions, err := de.model.Predict(boardState.Grid)
	inferDuration := time.Since(inferStart)

	de.mu.Lock()
	de.totalInferenceTime += inferDuration
	de.mu.Unlock()

	if err != nil {
		de.logger.Error("Inference failed", zap.Error(err))
		return nil, fmt.Errorf("inference failed: %w", err)
	}

	// Rank and select moves
	rankedMoves := de.rankMoves(predictions)

	if len(rankedMoves) == 0 {
		return nil, fmt.Errorf("no valid moves found")
	}

	// Check confidence threshold
	topMove := rankedMoves[0]
	if topMove.Confidence < de.confidenceThreshold {
		de.logger.Warn("Low confidence decision",
			zap.String("move", topMove.Move),
			zap.Float64("confidence", topMove.Confidence),
			zap.Float64("threshold", de.confidenceThreshold),
		)
		// Still return, but caller can decide whether to use it
	}

	// Build decision
	decision := &Decision{
		TopMove:      &topMove,
		Alternatives: rankedMoves[1:],
		Timestamp:    time.Now(),
		InferenceMs:  inferDuration.Seconds() * 1000,
		BoardState:   boardState.Grid,
		TotalMoves:   len(rankedMoves),
	}

	// Log decision
	de.logger.Info("Decision made",
		zap.String("move", topMove.Move),
		zap.Float64("confidence", topMove.Confidence),
		zap.Int("rank", topMove.Rank),
		zap.String("category", topMove.Category),
		zap.Float64("inference_ms", decision.InferenceMs),
		zap.Int("alternatives", len(decision.Alternatives)),
		zap.Duration("total_time", time.Since(startTime)),
	)

	return decision, nil
}

// captureBoardWithFallback attempts capture with retries
func (de *DecisionEngine) captureBoardWithFallback() (*vision.BoardState, error) {
	const maxRetries = 3
	const retryDelay = 100 * time.Millisecond

	var lastErr error

	for i := 0; i < maxRetries; i++ {
		boardState, err := de.capturer.ExtractBoardState()
		if err == nil {
			return boardState, nil
		}

		lastErr = err
		de.logger.Warn("Capture attempt failed",
			zap.Int("attempt", i+1),
			zap.Int("max_retries", maxRetries),
			zap.Error(err),
		)

		if i < maxRetries-1 {
			time.Sleep(retryDelay)
		}
	}

	return nil, fmt.Errorf("all capture attempts failed: %w", lastErr)
}

// rankMoves ranks predictions and returns top K moves with rich metadata and explanations
func (de *DecisionEngine) rankMoves(predictions []float64) []RankedMove {
	// Get top K moves from model
	topMoves := model.GetTopKMoves(predictions, de.topK)

	rankedMoves := make([]RankedMove, 0, len(topMoves))

	for i, moveScore := range topMoves {
		move := model.DecodeMove(moveScore.MoveIndex)

		// Detect patterns for this move
		patterns := de.detectMovePatterns(move, predictions)

		// Generate comprehensive explanation with pattern awareness
		explanation := de.generateRichExplanation(moveScore.Score, i+1, move, patterns)

		// Categorize with pattern consideration
		category := de.CategorizeMove(moveScore.Score, patterns)

		rankedMoves = append(rankedMoves, RankedMove{
			Move:        move,
			Confidence:  moveScore.Score,
			Rank:        i + 1,
			MoveIndex:   moveScore.MoveIndex,
			Explanation: explanation,
			Category:    category,
		})
	}

	return rankedMoves
}

// generateRichExplanation creates detailed, human-friendly move explanation with tactical context
func (de *DecisionEngine) generateRichExplanation(confidence float64, rank int, move string, patterns []string) string {
	explanation := ""

	// Primary assessment based on rank and confidence
	if rank == 1 {
		if confidence >= 0.85 {
			explanation = "🌟 Highly recommended. "
		} else if confidence >= 0.70 {
			explanation = "✓ Strong candidate. "
		} else if confidence >= 0.50 {
			explanation = "→ Reasonable choice. "
		} else {
			explanation = "⚠️ Uncertain position. "
		}
	} else {
		explanation = fmt.Sprintf("Alternative #%d: ", rank)
	}

	// Add tactical/strategic description from patterns
	if len(patterns) > 0 {
		primaryPattern := patterns[0]
		explanation += primaryPattern + ". "

		// Add secondary pattern if exists
		if len(patterns) > 1 && rank == 1 {
			explanation += patterns[1] + ". "
		}
	} else {
		// Generic descriptions based on confidence
		if confidence >= 0.70 {
			explanation += "Solid positional play. "
		} else if confidence >= 0.40 {
			explanation += "Developing the position. "
		} else {
			explanation += "Speculative move. "
		}
	}

	// Add confidence qualifier for top moves
	if rank == 1 {
		if confidence < de.confidenceThreshold {
			explanation += fmt.Sprintf("(Low confidence: %.1f%% < %.1f%% threshold) ",
				confidence*100, de.confidenceThreshold*100)
		} else if confidence >= 0.90 {
			explanation += fmt.Sprintf("(Very confident: %.1f%%) ", confidence*100)
		}
	}

	// Tactical hints for top move
	if rank == 1 && move != "" {
		tactical := de.detectTactical(move, nil)
		if tactical != "" {
			explanation += tactical + ". "
		}
	}

	return explanation
}

// categorizeMove categorizes a move based on confidence (legacy method kept for compatibility)
func (de *DecisionEngine) categorizeMove(confidence float64) string {
	switch {
	case confidence >= 0.80:
		return "Excellent"
	case confidence >= 0.60:
		return "Good"
	case confidence >= 0.40:
		return "Fair"
	case confidence >= 0.20:
		return "Risky"
	default:
		return "Uncertain"
	}
}

// explainMove generates explanation for a move (legacy method kept for compatibility)
func (de *DecisionEngine) explainMove(confidence float64, rank int) string {
	patterns := []string{}
	return de.generateRichExplanation(confidence, rank, "", patterns)
}

// GetStatistics returns engine statistics
func (de *DecisionEngine) GetStatistics() EngineStats {
	de.mu.RLock()
	defer de.mu.RUnlock()

	avgInferenceMs := 0.0
	if de.totalDecisions > 0 {
		avgInferenceMs = de.totalInferenceTime.Seconds() * 1000 / float64(de.totalDecisions)
	}

	captureRate := 0.0
	if de.totalDecisions > 0 {
		captureRate = float64(de.successfulCaptures) / float64(de.totalDecisions) * 100
	}

	return EngineStats{
		TotalDecisions:     de.totalDecisions,
		SuccessfulCaptures: de.successfulCaptures,
		FailedCaptures:     de.failedCaptures,
		CaptureSuccessRate: captureRate,
		AvgInferenceMs:     avgInferenceMs,
		TotalInferenceTime: de.totalInferenceTime,
	}
}

// EngineStats represents engine performance statistics
type EngineStats struct {
	TotalDecisions     int
	SuccessfulCaptures int
	FailedCaptures     int
	CaptureSuccessRate float64
	AvgInferenceMs     float64
	TotalInferenceTime time.Duration
}

// DecisionWithContext makes a decision from a provided board state
func (de *DecisionEngine) DecisionWithContext(boardState []float64) (*Decision, error) {
	startTime := time.Now()

	de.mu.Lock()
	de.totalDecisions++
	de.mu.Unlock()

	// Run inference
	inferStart := time.Now()
	predictions, err := de.model.Predict(boardState)
	inferDuration := time.Since(inferStart)

	de.mu.Lock()
	de.totalInferenceTime += inferDuration
	de.mu.Unlock()

	if err != nil {
		de.logger.Error("Inference failed", zap.Error(err))
		return nil, fmt.Errorf("inference failed: %w", err)
	}

	// Rank moves
	rankedMoves := de.rankMoves(predictions)

	if len(rankedMoves) == 0 {
		return nil, fmt.Errorf("no valid moves found")
	}

	topMove := rankedMoves[0]

	decision := &Decision{
		TopMove:      &topMove,
		Alternatives: rankedMoves[1:],
		Timestamp:    time.Now(),
		InferenceMs:  inferDuration.Seconds() * 1000,
		BoardState:   boardState,
		TotalMoves:   len(rankedMoves),
	}

	de.logger.Info("Decision from context",
		zap.String("move", topMove.Move),
		zap.Float64("confidence", topMove.Confidence),
		zap.Float64("inference_ms", decision.InferenceMs),
		zap.Duration("total_time", time.Since(startTime)),
	)

	return decision, nil
}

// BatchDecisions makes decisions on multiple board states
func (de *DecisionEngine) BatchDecisions(boardStates [][]float64) ([]*Decision, error) {
	decisions := make([]*Decision, 0, len(boardStates))

	for i, state := range boardStates {
		decision, err := de.DecisionWithContext(state)
		if err != nil {
			de.logger.Warn("Batch decision failed",
				zap.Int("index", i),
				zap.Error(err),
			)
			continue
		}
		decisions = append(decisions, decision)
	}

	if len(decisions) == 0 {
		return nil, fmt.Errorf("all batch decisions failed")
	}

	return decisions, nil
}

// CompareDecisions compares two decisions for analysis
func CompareDecisions(d1, d2 *Decision) *DecisionComparison {
	return &DecisionComparison{
		Decision1:       d1,
		Decision2:       d2,
		SameTopMove:     d1.TopMove.Move == d2.TopMove.Move,
		ConfidenceDelta: d1.TopMove.Confidence - d2.TopMove.Confidence,
		InferenceDelta:  d1.InferenceMs - d2.InferenceMs,
		TimeDelta:       d2.Timestamp.Sub(d1.Timestamp),
	}
}

// DecisionComparison represents comparison between two decisions
type DecisionComparison struct {
	Decision1       *Decision
	Decision2       *Decision
	SameTopMove     bool
	ConfidenceDelta float64
	InferenceDelta  float64
	TimeDelta       time.Duration
}

// DecisionHistory tracks decision history with analysis
type DecisionHistory struct {
	decisions []Decision
	maxSize   int
	mu        sync.RWMutex
}

// NewDecisionHistory creates a new decision history tracker
func NewDecisionHistory(maxSize int) *DecisionHistory {
	return &DecisionHistory{
		decisions: make([]Decision, 0, maxSize),
		maxSize:   maxSize,
	}
}

// Add adds a decision to history
func (dh *DecisionHistory) Add(decision *Decision) {
	dh.mu.Lock()
	defer dh.mu.Unlock()

	dh.decisions = append(dh.decisions, *decision)
	if len(dh.decisions) > dh.maxSize {
		dh.decisions = dh.decisions[1:]
	}
}

// GetRecent returns N most recent decisions
func (dh *DecisionHistory) GetRecent(n int) []Decision {
	dh.mu.RLock()
	defer dh.mu.RUnlock()

	if n > len(dh.decisions) {
		n = len(dh.decisions)
	}

	start := len(dh.decisions) - n
	recent := make([]Decision, n)
	copy(recent, dh.decisions[start:])
	return recent
}

// GetStats returns statistics about decision history
func (dh *DecisionHistory) GetStats() HistoryStats {
	dh.mu.RLock()
	defer dh.mu.RUnlock()

	if len(dh.decisions) == 0 {
		return HistoryStats{}
	}

	var totalConfidence float64
	var totalInferenceMs float64
	categoryCounts := make(map[string]int)

	for _, decision := range dh.decisions {
		totalConfidence += decision.TopMove.Confidence
		totalInferenceMs += decision.InferenceMs
		categoryCounts[decision.TopMove.Category]++
	}

	return HistoryStats{
		TotalDecisions: len(dh.decisions),
		AvgConfidence:  totalConfidence / float64(len(dh.decisions)),
		AvgInferenceMs: totalInferenceMs / float64(len(dh.decisions)),
		CategoryCounts: categoryCounts,
	}
}

// HistoryStats represents statistics about decision history
type HistoryStats struct {
	TotalDecisions int
	AvgConfidence  float64
	AvgInferenceMs float64
	CategoryCounts map[string]int
}

// MoveFormatter formats moves for display
type MoveFormatter struct{}

// FormatMove formats a move in human-readable format
func (mf *MoveFormatter) FormatMove(move string) string {
	if len(move) != 4 {
		return move
	}

	from := move[0:2]
	to := move[2:4]

	return fmt.Sprintf("Move from %s to %s", from, to)
}

// FormatMoveWithPiece formats move with piece information with rich context
func (mf *MoveFormatter) FormatMoveWithPiece(move string, pieceType string) string {
	if pieceType == "" {
		return mf.FormatMove(move)
	}

	if len(move) != 4 {
		return move
	}

	from := move[0:2]
	to := move[2:4]

	// Map piece abbreviations to names
	pieceNames := map[string]string{
		"P": "Pawn", "N": "Knight", "B": "Bishop",
		"R": "Rook", "Q": "Queen", "K": "King",
		"p": "pawn", "n": "knight", "b": "bishop",
		"r": "rook", "q": "queen", "k": "king",
	}

	pieceName := pieceNames[pieceType]
	if pieceName == "" {
		pieceName = pieceType
	}

	return fmt.Sprintf("%s: %s → %s", pieceName, from, to)
}

// GenerateMoveExplanation creates a detailed explanation for a move
func (de *DecisionEngine) GenerateMoveExplanation(move RankedMove, boardState []float64) string {
	explanation := ""

	// Confidence-based description
	if move.Confidence > 0.9 {
		explanation = "Strong tactical move. "
	} else if move.Confidence > 0.7 {
		explanation = "Solid positional move. "
	} else if move.Confidence > 0.5 {
		explanation = "Reasonable option. "
	} else {
		explanation = "Speculative move. "
	}

	// Detect move patterns
	patterns := de.detectMovePatterns(move.Move, boardState)
	if len(patterns) > 0 {
		explanation += fmt.Sprintf("Pattern: %s. ", patterns[0])
	}

	// Tactical hints
	tactical := de.detectTactical(move.Move, boardState)
	if tactical != "" {
		explanation += tactical + " "
	}

	// Confidence qualifier
	if move.Confidence < de.confidenceThreshold {
		explanation += "⚠️ Below confidence threshold. "
	}

	return explanation
}

// detectMovePatterns identifies common chess patterns in a move
func (de *DecisionEngine) detectMovePatterns(move string, boardState []float64) []string {
	if len(move) != 4 {
		return nil
	}

	patterns := []string{}

	fromFile := int(move[0] - 'a')
	fromRank := int(move[1] - '1')
	toFile := int(move[2] - 'a')
	toRank := int(move[3] - '1')

	fileDiff := abs(toFile - fromFile)
	rankDiff := abs(toRank - fromRank)

	// Detect move types
	if fileDiff == 0 {
		patterns = append(patterns, "Vertical advance")
	} else if rankDiff == 0 {
		patterns = append(patterns, "Lateral maneuver")
	} else if fileDiff == rankDiff {
		patterns = append(patterns, "Diagonal thrust")
	}

	// Detect special squares
	centerSquares := map[string]bool{
		"d4": true, "d5": true, "e4": true, "e5": true,
	}
	toSquare := move[2:4]
	if centerSquares[toSquare] {
		patterns = append(patterns, "Controls center")
	}

	// Detect advancing moves
	if (fromRank < 4 && toRank >= 4) || (fromRank >= 4 && toRank < 4) {
		patterns = append(patterns, "Advances position")
	}

	// Detect knight patterns
	if (rankDiff == 2 && fileDiff == 1) || (rankDiff == 1 && fileDiff == 2) {
		patterns = append(patterns, "Knight fork potential")
	}

	// Detect castling
	if fromFile == 4 && (toFile == 2 || toFile == 6) { // King moving 2 squares
		if fromRank == 0 || fromRank == 7 {
			if toFile == 2 {
				patterns = append(patterns, "Queenside castling")
			} else {
				patterns = append(patterns, "Kingside castling")
			}
		}
	}

	// Detect pawn advances
	if fileDiff == 0 && rankDiff == 2 && (fromRank == 1 || fromRank == 6) {
		patterns = append(patterns, "Pawn opening")
	}

	// Detect promotion squares
	if toRank == 0 || toRank == 7 {
		patterns = append(patterns, "Promotion threat")
	}

	return patterns
}

// detectTactical identifies tactical motifs
func (de *DecisionEngine) detectTactical(move string, boardState []float64) string {
	if len(move) != 4 {
		return ""
	}

	toFile := int(move[2] - 'a')
	toRank := int(move[3] - '1')

	// Simple tactical hints based on destination
	// (In production, analyze actual board state for captures, pins, forks, etc.)

	// Check if moving to enemy territory
	if toRank >= 5 { // White advancing deep
		return "Aggressive advance into enemy territory"
	} else if toRank <= 2 { // Black advancing deep
		return "Defensive consolidation"
	}

	// Check edge moves (potentially defensive)
	if toFile == 0 || toFile == 7 {
		return "Flank operation"
	}

	return ""
}

// CategorizeMove assigns a category based on confidence and patterns
func (de *DecisionEngine) CategorizeMove(confidence float64, patterns []string) string {
	hasStrongPattern := false
	for _, pattern := range patterns {
		if pattern == "Controls center" || pattern == "Knight fork potential" || pattern == "Castling" {
			hasStrongPattern = true
			break
		}
	}

	if confidence >= 0.9 {
		return "Excellent"
	} else if confidence >= 0.7 {
		if hasStrongPattern {
			return "Good"
		}
		return "Solid"
	} else if confidence >= 0.5 {
		return "Fair"
	} else if confidence >= 0.3 {
		return "Risky"
	} else {
		return "Speculative"
	}
}

// Helper function
func abs(x int) int {
	if x < 0 {
		return -x
	}
	return x
}

// FormatDecision formats a complete decision for display
func (mf *MoveFormatter) FormatDecision(decision *Decision) string {
	result := fmt.Sprintf("Top Move: %s\n", mf.FormatMove(decision.TopMove.Move))
	result += fmt.Sprintf("Confidence: %.1f%% (%s)\n",
		decision.TopMove.Confidence*100,
		decision.TopMove.Category)
	result += fmt.Sprintf("Explanation: %s\n", decision.TopMove.Explanation)

	if len(decision.Alternatives) > 0 {
		result += "\nAlternatives:\n"
		for i, alt := range decision.Alternatives {
			if i >= 3 {
				break
			}
			result += fmt.Sprintf("  %d. %s (%.1f%% - %s)\n",
				alt.Rank,
				mf.FormatMove(alt.Move),
				alt.Confidence*100,
				alt.Category)
		}
	}

	result += fmt.Sprintf("\nInference Time: %.2f ms\n", decision.InferenceMs)

	return result
}
