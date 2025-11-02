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

// ChessFeatures contains extracted chess-specific features
type ChessFeatures struct {
	// Material counts (normalized 0-1)
	WhitePawns   float32
	WhiteKnights float32
	WhiteBishops float32
	WhiteRooks   float32
	WhiteQueens  float32
	BlackPawns   float32
	BlackKnights float32
	BlackBishops float32
	BlackRooks   float32
	BlackQueens  float32

	// Derived features
	MaterialBalance      float32 // Positive = white advantage
	WhiteKingSafety      float32 // 0 = safe, 1 = exposed
	BlackKingSafety      float32
	WhitePawnAdvancement float32
	BlackPawnAdvancement float32
	WhiteCenterControl   float32
	BlackCenterControl   float32
	WhiteMobility        float32
	BlackMobility        float32
	GamePhase            float32 // 0 = opening, 1 = endgame
}

// ExtractChessFeatures extracts chess-specific features from board tensor
func ExtractChessFeatures(tensor [12][8][8]float32) ChessFeatures {
	var features ChessFeatures

	// Count pieces
	var whitePawns, blackPawns float32
	var whiteKnights, blackKnights float32
	var whiteBishops, blackBishops float32
	var whiteRooks, blackRooks float32
	var whiteQueens, blackQueens float32
	var whiteKingX, whiteKingY int = -1, -1
	var blackKingX, blackKingY int = -1, -1

	for r := 0; r < 8; r++ {
		for c := 0; c < 8; c++ {
			// White pieces (channels 0-5)
			if tensor[0][r][c] > 0 {
				whitePawns++
			}
			if tensor[1][r][c] > 0 {
				whiteKnights++
			}
			if tensor[2][r][c] > 0 {
				whiteBishops++
			}
			if tensor[3][r][c] > 0 {
				whiteRooks++
			}
			if tensor[4][r][c] > 0 {
				whiteQueens++
			}
			if tensor[5][r][c] > 0 {
				whiteKingX, whiteKingY = c, r
			}

			// Black pieces (channels 6-11)
			if tensor[6][r][c] > 0 {
				blackPawns++
			}
			if tensor[7][r][c] > 0 {
				blackKnights++
			}
			if tensor[8][r][c] > 0 {
				blackBishops++
			}
			if tensor[9][r][c] > 0 {
				blackRooks++
			}
			if tensor[10][r][c] > 0 {
				blackQueens++
			}
			if tensor[11][r][c] > 0 {
				blackKingX, blackKingY = c, r
			}
		}
	}

	// Normalize piece counts
	features.WhitePawns = whitePawns / 8.0
	features.WhiteKnights = whiteKnights / 2.0
	features.WhiteBishops = whiteBishops / 2.0
	features.WhiteRooks = whiteRooks / 2.0
	features.WhiteQueens = whiteQueens
	features.BlackPawns = blackPawns / 8.0
	features.BlackKnights = blackKnights / 2.0
	features.BlackBishops = blackBishops / 2.0
	features.BlackRooks = blackRooks / 2.0
	features.BlackQueens = blackQueens

	// Calculate material balance (standard piece values)
	whiteMaterial := whitePawns + whiteKnights*3 + whiteBishops*3 + whiteRooks*5 + whiteQueens*9
	blackMaterial := blackPawns + blackKnights*3 + blackBishops*3 + blackRooks*5 + blackQueens*9
	totalMaterial := whiteMaterial + blackMaterial

	if totalMaterial > 0 {
		features.MaterialBalance = (whiteMaterial - blackMaterial) / 39.0 // Normalize by starting material
	}

	// King safety (distance from center - risky in middlegame, ok in endgame)
	if whiteKingX >= 0 {
		centerDist := float32(absInt(whiteKingX-3) + absInt(whiteKingY-3))
		features.WhiteKingSafety = centerDist / 6.0 // Max distance from center
	}
	if blackKingX >= 0 {
		centerDist := float32(absInt(blackKingX-3) + absInt(blackKingY-3))
		features.BlackKingSafety = centerDist / 6.0
	}

	// Pawn advancement (higher rank = more advanced)
	var whiteAdvancement, blackAdvancement float32
	for c := 0; c < 8; c++ {
		for r := 0; r < 8; r++ {
			if tensor[0][r][c] > 0 { // White pawn
				whiteAdvancement += float32(r) / 7.0
			}
			if tensor[6][r][c] > 0 { // Black pawn
				blackAdvancement += float32(7-r) / 7.0
			}
		}
	}
	if whitePawns > 0 {
		features.WhitePawnAdvancement = whiteAdvancement / whitePawns
	}
	if blackPawns > 0 {
		features.BlackPawnAdvancement = blackAdvancement / blackPawns
	}

	// Center control (d4, d5, e4, e5)
	var whiteCenter, blackCenter float32
	centerSquares := [][2]int{{3, 3}, {3, 4}, {4, 3}, {4, 4}}
	for _, sq := range centerSquares {
		r, c := sq[0], sq[1]
		for ch := 0; ch < 6; ch++ { // White pieces
			if tensor[ch][r][c] > 0 {
				whiteCenter++
			}
		}
		for ch := 6; ch < 12; ch++ { // Black pieces
			if tensor[ch][r][c] > 0 {
				blackCenter++
			}
		}
	}
	features.WhiteCenterControl = whiteCenter / 4.0
	features.BlackCenterControl = blackCenter / 4.0

	// Mobility (rough estimate based on piece types and positions)
	features.WhiteMobility = calculateMobilityEstimate(tensor, true) / 100.0
	features.BlackMobility = calculateMobilityEstimate(tensor, false) / 100.0

	// Game phase (0 = opening/middlegame, 1 = endgame)
	if totalMaterial > 0 {
		features.GamePhase = 1.0 - (totalMaterial / 78.0) // 78 = starting material
	}

	return features
}

// calculateMobilityEstimate estimates piece mobility
func calculateMobilityEstimate(tensor [12][8][8]float32, isWhite bool) float32 {
	mobility := float32(0.0)
	startCh, endCh := 0, 6
	if !isWhite {
		startCh, endCh = 6, 12
	}

	for ch := startCh; ch < endCh; ch++ {
		pieceType := ch % 6
		for r := 0; r < 8; r++ {
			for c := 0; c < 8; c++ {
				if tensor[ch][r][c] > 0 {
					// Approximate mobility by piece type
					switch pieceType {
					case 0: // Pawn
						mobility += 1.0
					case 1: // Knight
						mobility += 4.0
					case 2: // Bishop
						mobility += 7.0
					case 3: // Rook
						mobility += 10.0
					case 4: // Queen
						mobility += 15.0
					case 5: // King
						mobility += 3.0
					}

					// Bonus for central pieces
					if c >= 2 && c <= 5 && r >= 2 && r <= 5 {
						mobility += 1.0
					}
				}
			}
		}
	}

	return mobility
}

// absInt returns absolute value of int
func absInt(x int) int {
	if x < 0 {
		return -x
	}
	return x
}

// absFloat returns absolute value of float32
func absFloat(x float32) float32 {
	if x < 0 {
		return -x
	}
	return x
}

// IsLegalMovePlausible checks if a move is plausibly legal (basic check)
func IsLegalMovePlausible(fromSquare, toSquare int, tensor [12][8][8]float32) bool {
	if fromSquare < 0 || fromSquare >= 64 || toSquare < 0 || toSquare >= 64 {
		return false
	}

	fromRow, fromCol := fromSquare/8, fromSquare%8
	toRow, toCol := toSquare/8, toSquare%8

	// Check if source has a piece
	hasPiece := false
	movingPieceType := -1
	isWhite := false

	for ch := 0; ch < 12; ch++ {
		if tensor[ch][fromRow][fromCol] > 0 {
			hasPiece = true
			movingPieceType = ch % 6
			isWhite = ch < 6
			break
		}
	}

	if !hasPiece {
		return false
	}

	// Check if capturing own piece
	targetStart, targetEnd := 0, 6
	if !isWhite {
		targetStart, targetEnd = 6, 12
	}

	for ch := targetStart; ch < targetEnd; ch++ {
		if tensor[ch][toRow][toCol] > 0 {
			return false // Can't capture own piece
		}
	}

	// Basic movement rules
	rowDiff := absInt(toRow - fromRow)
	colDiff := absInt(toCol - fromCol)

	switch movingPieceType {
	case 0: // Pawn
		// Very basic pawn check (no en passant, promotion complexity)
		if isWhite {
			if toRow <= fromRow {
				return false // Must move forward
			}
			if colDiff == 0 && rowDiff <= 2 {
				return true // Forward move (simplified)
			}
			if colDiff == 1 && rowDiff == 1 {
				return true // Diagonal capture (simplified)
			}
		} else {
			if toRow >= fromRow {
				return false
			}
			if colDiff == 0 && rowDiff <= 2 {
				return true
			}
			if colDiff == 1 && rowDiff == 1 {
				return true
			}
		}
		return false

	case 1: // Knight
		return (rowDiff == 2 && colDiff == 1) || (rowDiff == 1 && colDiff == 2)

	case 2: // Bishop
		return rowDiff == colDiff && rowDiff > 0

	case 3: // Rook
		return (rowDiff == 0 && colDiff > 0) || (colDiff == 0 && rowDiff > 0)

	case 4: // Queen
		return (rowDiff == colDiff && rowDiff > 0) ||
			(rowDiff == 0 && colDiff > 0) ||
			(colDiff == 0 && rowDiff > 0)

	case 5: // King
		return rowDiff <= 1 && colDiff <= 1 && (rowDiff+colDiff) > 0
	}

	return true
}

// FilterIllegalMoves removes obviously illegal moves from predictions
func FilterIllegalMoves(predictions []MovePrediction, tensor [12][8][8]float32) []MovePrediction {
	filtered := make([]MovePrediction, 0, len(predictions))

	for _, pred := range predictions {
		if IsLegalMovePlausible(pred.FromSquare, pred.ToSquare, tensor) {
			filtered = append(filtered, pred)
		}
	}

	return filtered
}
