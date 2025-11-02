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

// rankMoves ranks predictions and returns top K moves with metadata
func (de *DecisionEngine) rankMoves(predictions []float64) []RankedMove {
	// Get top K moves from model
	topMoves := model.GetTopKMoves(predictions, de.topK)

	rankedMoves := make([]RankedMove, 0, len(topMoves))

	for i, moveScore := range topMoves {
		move := model.DecodeMove(moveScore.MoveIndex)
		category := de.categorizeMove(moveScore.Score)
		explanation := de.explainMove(moveScore.Score, i+1)

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

// categorizeMove categorizes a move based on confidence
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

// explainMove generates explanation for a move
func (de *DecisionEngine) explainMove(confidence float64, rank int) string {
	switch {
	case rank == 1 && confidence >= 0.80:
		return "Strong move with high confidence"
	case rank == 1 && confidence >= 0.60:
		return "Solid move, recommended"
	case rank == 1 && confidence >= 0.40:
		return "Moderate choice, consider alternatives"
	case rank == 1:
		return "Low confidence, position may be complex"
	case rank <= 3:
		return fmt.Sprintf("Alternative #%d - viable option", rank)
	default:
		return fmt.Sprintf("Lower priority option (rank #%d)", rank)
	}
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

// FormatMoveWithPiece formats move with piece information (placeholder)
func (mf *MoveFormatter) FormatMoveWithPiece(move string, pieceType string) string {
	if pieceType == "" {
		return mf.FormatMove(move)
	}

	if len(move) != 4 {
		return move
	}

	to := move[2:4]
	return fmt.Sprintf("Move %s to %s", pieceType, to)
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

// ============================================================================
// COMPATIBILITY LAYER - Legacy API support for cmd/partner/main.go
// ============================================================================

// Prediction represents a simple prediction (legacy compatibility)
type Prediction struct {
	Move       string
	Confidence float64
	Timestamp  time.Time
}

// Advisor provides advice with history tracking (legacy compatibility)
type Advisor struct {
	engine     *DecisionEngine
	history    *DecisionHistory
	lastAdvice *Advice
	mu         sync.RWMutex
}

// Advice represents advice with alternatives (legacy compatibility)
type Advice struct {
	PrimaryMove  string
	Alternatives []string
	Confidence   float64
	Timestamp    time.Time
	Explanation  string
}

// NewEngine creates a basic decision engine (legacy compatibility)
// This is a wrapper around NewDecisionEngine with default parameters
func NewEngine(net *model.ChessNet, capturer *vision.Capturer, confidenceThreshold float64) *DecisionEngine {
	logger, _ := zap.NewProduction()
	return NewDecisionEngine(net, capturer, confidenceThreshold, 5, logger)
}

// NewAdvisor creates an advisor with history (legacy compatibility)
func NewAdvisor(engine *DecisionEngine, historySize int) *Advisor {
	return &Advisor{
		engine:  engine,
		history: NewDecisionHistory(historySize),
	}
}

// GetAdvice returns advice based on current board state (legacy compatibility)
func (a *Advisor) GetAdvice() (*Advice, error) {
	a.mu.Lock()
	defer a.mu.Unlock()

	decision, err := a.engine.MakeDecision()
	if err != nil {
		return nil, err
	}

	// Store in history
	a.history.Add(decision)

	// Build alternatives list
	alternatives := make([]string, 0, len(decision.Alternatives))
	for _, alt := range decision.Alternatives {
		if len(alternatives) >= 3 {
			break
		}
		alternatives = append(alternatives, alt.Move)
	}

	advice := &Advice{
		PrimaryMove:  decision.TopMove.Move,
		Alternatives: alternatives,
		Confidence:   decision.TopMove.Confidence,
		Timestamp:    decision.Timestamp,
		Explanation:  decision.TopMove.Explanation,
	}

	a.lastAdvice = advice
	return advice, nil
}

// GetLastAdvice returns the most recent advice (legacy compatibility)
func (a *Advisor) GetLastAdvice() *Advice {
	a.mu.RLock()
	defer a.mu.RUnlock()
	return a.lastAdvice
}

// GetHistory returns the decision history (legacy compatibility)
func (a *Advisor) GetHistory() *DecisionHistory {
	return a.history
}

// Statistics represents system statistics (legacy compatibility)
type Statistics struct {
	TotalPredictions   int
	AverageConfidence  float64
	SuccessfulCaptures int
	FailedCaptures     int
	AverageInferenceMs float64
}

// FormatAdvice formats advice for display (legacy compatibility)
func FormatAdvice(advice *Advice) string {
	result := fmt.Sprintf("Primary Move: %s\n", advice.PrimaryMove)
	result += fmt.Sprintf("Confidence: %.1f%%\n", advice.Confidence*100)
	
	if advice.Explanation != "" {
		result += fmt.Sprintf("Explanation: %s\n", advice.Explanation)
	}
	
	if len(advice.Alternatives) > 0 {
		result += "\nAlternatives:\n"
		for i, alt := range advice.Alternatives {
			result += fmt.Sprintf("  %d. %s\n", i+1, alt)
		}
	}
	
	return result
}
