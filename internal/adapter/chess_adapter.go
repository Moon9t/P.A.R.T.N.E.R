package adapter

import (
	"fmt"
	"strings"

	"gorgonia.org/tensor"
)

// ChessAdapter implements GameAdapter for chess
type ChessAdapter struct {
	*BaseAdapter
	lastState  [12][8][8]float32
	lastAction interface{}
}

// NewChessAdapter creates a new chess adapter
func NewChessAdapter() *ChessAdapter {
	// Chess state: 12 channels (6 piece types × 2 colors) × 8×8 board
	stateDims := []int{12, 8, 8}
	// Chess action: 4096 possible moves (64 from × 64 to squares)
	actionDims := []int{4096}

	return &ChessAdapter{
		BaseAdapter: NewBaseAdapter("chess", stateDims, actionDims),
	}
}

// EncodeState converts chess board state to tensor
// Accepts: [12][8][8]float32 (piece planes), FEN string, or map[string]interface{}
func (c *ChessAdapter) EncodeState(frame interface{}) (tensor.Tensor, error) {
	switch state := frame.(type) {
	case [12][8][8]float32:
		// Already in correct format
		return c.encodeBoardTensor(state)

	case string:
		// FEN string
		board, err := c.parseFEN(state)
		if err != nil {
			return nil, fmt.Errorf("failed to parse FEN: %w", err)
		}
		return c.encodeBoardTensor(board)

	case map[string]interface{}:
		// Generic board representation
		board, err := c.parseGenericBoard(state)
		if err != nil {
			return nil, fmt.Errorf("failed to parse board: %w", err)
		}
		return c.encodeBoardTensor(board)

	default:
		return nil, fmt.Errorf("unsupported state type: %T", frame)
	}
}

// EncodeAction converts chess move to tensor
// Accepts: string ("e2e4"), map with from/to, or struct with FromSquare/ToSquare
func (c *ChessAdapter) EncodeAction(action interface{}) (tensor.Tensor, error) {
	var fromSquare, toSquare int
	var err error

	switch act := action.(type) {
	case string:
		// Parse move string like "e2e4"
		fromSquare, toSquare, err = c.parseMoveString(act)
		if err != nil {
			return nil, err
		}

	case map[string]interface{}:
		// Parse from map
		from, okFrom := act["from"].(int)
		to, okTo := act["to"].(int)
		if !okFrom || !okTo {
			return nil, fmt.Errorf("action map must contain 'from' and 'to' integers")
		}
		fromSquare, toSquare = from, to

	case struct {
		FromSquare int
		ToSquare   int
	}:
		fromSquare = act.FromSquare
		toSquare = act.ToSquare

	default:
		return nil, fmt.Errorf("unsupported action type: %T", action)
	}

	// Validate squares
	if fromSquare < 0 || fromSquare >= 64 || toSquare < 0 || toSquare >= 64 {
		return nil, fmt.Errorf("invalid squares: from=%d, to=%d", fromSquare, toSquare)
	}

	// Encode as one-hot vector
	moveIndex := fromSquare*64 + toSquare
	data := make([]float64, 4096)
	data[moveIndex] = 1.0

	return tensor.New(
		tensor.WithShape(4096),
		tensor.WithBacking(data),
	), nil
}

// DecodeAction converts network prediction to chess move
func (c *ChessAdapter) DecodeAction(pred tensor.Tensor) (interface{}, error) {
	// Pred should be a 4096-dim probability distribution
	shape := pred.Shape()
	if len(shape) != 1 || shape[0] != 4096 {
		return nil, fmt.Errorf("invalid prediction shape: %v (expected [4096])", shape)
	}

	// Get data
	data := pred.Data().([]float64)

	// Find highest probability move
	maxIdx := 0
	maxProb := data[0]
	for i := 1; i < len(data); i++ {
		if data[i] > maxProb {
			maxProb = data[i]
			maxIdx = i
		}
	}

	// Decode move index
	fromSquare := maxIdx / 64
	toSquare := maxIdx % 64

	// Convert to algebraic notation
	move := squareToAlgebraic(fromSquare) + squareToAlgebraic(toSquare)

	return map[string]interface{}{
		"move":        move,
		"from_square": fromSquare,
		"to_square":   toSquare,
		"probability": maxProb,
	}, nil
}

// Feedback stores the correct action for learning
func (c *ChessAdapter) Feedback(correctAction interface{}) error {
	// Encode the correct action
	actionTensor, err := c.EncodeAction(correctAction)
	if err != nil {
		return fmt.Errorf("failed to encode correct action: %w", err)
	}

	// Store as experience if we have a previous state
	if c.lastAction != nil {
		stateTensor, err := c.encodeBoardTensor(c.lastState)
		if err != nil {
			return fmt.Errorf("failed to encode state: %w", err)
		}

		exp := Experience{
			State:    stateTensor,
			Action:   actionTensor,
			Reward:   1.0, // Correct move gets positive reward
			Done:     false,
			Metadata: map[string]interface{}{"correct_action": correctAction},
		}

		c.AddExperience(exp)
	}

	return nil
}

// ValidateState checks if chess state is valid
func (c *ChessAdapter) ValidateState(frame interface{}) error {
	switch state := frame.(type) {
	case [12][8][8]float32:
		// Check that we have exactly one king per side
		whiteKings := 0
		blackKings := 0
		for r := 0; r < 8; r++ {
			for f := 0; f < 8; f++ {
				if state[5][r][f] > 0 {
					whiteKings++
				}
				if state[11][r][f] > 0 {
					blackKings++
				}
			}
		}
		if whiteKings != 1 || blackKings != 1 {
			return fmt.Errorf("invalid board: white kings=%d, black kings=%d", whiteKings, blackKings)
		}
		return nil

	case string:
		// Basic FEN validation
		if !strings.Contains(state, "/") {
			return fmt.Errorf("invalid FEN format")
		}
		return nil

	default:
		return fmt.Errorf("unsupported state type for validation: %T", frame)
	}
}

// Helper functions

func (c *ChessAdapter) encodeBoardTensor(board [12][8][8]float32) (tensor.Tensor, error) {
	// Store for feedback
	c.lastState = board

	// Convert to flat array
	data := make([]float64, 12*8*8)
	idx := 0
	for ch := 0; ch < 12; ch++ {
		for r := 0; r < 8; r++ {
			for f := 0; f < 8; f++ {
				data[idx] = float64(board[ch][r][f])
				idx++
			}
		}
	}

	return tensor.New(
		tensor.WithShape(12, 8, 8),
		tensor.WithBacking(data),
	), nil
}

func (c *ChessAdapter) parseFEN(fen string) ([12][8][8]float32, error) {
	var board [12][8][8]float32

	// Split FEN (we only care about piece placement)
	parts := strings.Fields(fen)
	if len(parts) == 0 {
		return board, fmt.Errorf("empty FEN string")
	}

	placement := parts[0]
	ranks := strings.Split(placement, "/")
	if len(ranks) != 8 {
		return board, fmt.Errorf("FEN must have 8 ranks")
	}

	// Map FEN characters to piece indices
	// White pieces: P=0, N=1, B=2, R=3, Q=4, K=5
	// Black pieces: p=6, n=7, b=8, r=9, q=10, k=11
	pieceMap := map[rune]int{
		'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
		'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11,
	}

	for r, rank := range ranks {
		file := 0
		for _, char := range rank {
			if char >= '1' && char <= '8' {
				// Empty squares
				file += int(char - '0')
			} else if pieceIdx, ok := pieceMap[char]; ok {
				// Place piece
				board[pieceIdx][r][file] = 1.0
				file++
			}
		}
	}

	return board, nil
}

func (c *ChessAdapter) parseGenericBoard(data map[string]interface{}) ([12][8][8]float32, error) {
	var board [12][8][8]float32

	// Expect format: {"pieces": [[piece, row, col], ...]}
	pieces, ok := data["pieces"].([]interface{})
	if !ok {
		return board, fmt.Errorf("board must contain 'pieces' array")
	}

	for _, p := range pieces {
		piece, ok := p.(map[string]interface{})
		if !ok {
			continue
		}

		pieceType, _ := piece["type"].(string)
		row, _ := piece["row"].(int)
		col, _ := piece["col"].(int)
		color, _ := piece["color"].(string)

		if row < 0 || row >= 8 || col < 0 || col >= 8 {
			continue
		}

		// Map to channel
		channel := c.pieceTypeToChannel(pieceType, color)
		if channel >= 0 && channel < 12 {
			board[channel][row][col] = 1.0
		}
	}

	return board, nil
}

func (c *ChessAdapter) pieceTypeToChannel(pieceType, color string) int {
	baseChannel := 0
	if color == "black" {
		baseChannel = 6
	}

	switch strings.ToLower(pieceType) {
	case "pawn", "p":
		return baseChannel + 0
	case "knight", "n":
		return baseChannel + 1
	case "bishop", "b":
		return baseChannel + 2
	case "rook", "r":
		return baseChannel + 3
	case "queen", "q":
		return baseChannel + 4
	case "king", "k":
		return baseChannel + 5
	default:
		return -1
	}
}

func (c *ChessAdapter) parseMoveString(move string) (int, int, error) {
	// Parse moves like "e2e4" or "e2-e4"
	move = strings.ReplaceAll(move, "-", "")
	move = strings.ToLower(move)

	if len(move) < 4 {
		return 0, 0, fmt.Errorf("move string too short: %s", move)
	}

	from := move[0:2]
	to := move[2:4]

	fromSquare, err := algebraicToSquare(from)
	if err != nil {
		return 0, 0, fmt.Errorf("invalid from square: %w", err)
	}

	toSquare, err := algebraicToSquare(to)
	if err != nil {
		return 0, 0, fmt.Errorf("invalid to square: %w", err)
	}

	return fromSquare, toSquare, nil
}

// Helper: convert algebraic notation to square index (0-63)
func algebraicToSquare(sq string) (int, error) {
	if len(sq) != 2 {
		return 0, fmt.Errorf("invalid square: %s", sq)
	}

	file := int(sq[0] - 'a')
	rank := int(sq[1] - '1')

	if file < 0 || file >= 8 || rank < 0 || rank >= 8 {
		return 0, fmt.Errorf("square out of bounds: %s", sq)
	}

	return rank*8 + file, nil
}

// Helper: convert square index to algebraic notation
func squareToAlgebraic(square int) string {
	if square < 0 || square >= 64 {
		return "??"
	}

	rank := square / 8
	file := square % 8

	return string(rune('a'+file)) + string(rune('1'+rank))
}

// ===== Advanced Chess Features =====

// ChessPosition represents a complete chess position with full game state
type ChessPosition struct {
	Board           [12][8][8]float32
	WhiteToMove     bool
	CastlingRights  CastlingRights
	EnPassantSquare int // -1 if none
	HalfMoveClock   int
	FullMoveNumber  int
}

// CastlingRights tracks castling availability
type CastlingRights struct {
	WhiteKingSide  bool
	WhiteQueenSide bool
	BlackKingSide  bool
	BlackQueenSide bool
}

// ParseFullFEN parses a complete FEN string including all game state
func (c *ChessAdapter) ParseFullFEN(fen string) (*ChessPosition, error) {
	parts := strings.Fields(fen)
	if len(parts) < 4 {
		return nil, fmt.Errorf("incomplete FEN string: %s", fen)
	}

	pos := &ChessPosition{
		EnPassantSquare: -1,
	}

	// Parse piece placement
	board, err := c.parseFEN(fen)
	if err != nil {
		return nil, fmt.Errorf("failed to parse board: %w", err)
	}
	pos.Board = board

	// Parse active color
	pos.WhiteToMove = (parts[1] == "w")

	// Parse castling rights
	castling := parts[2]
	pos.CastlingRights = CastlingRights{
		WhiteKingSide:  strings.Contains(castling, "K"),
		WhiteQueenSide: strings.Contains(castling, "Q"),
		BlackKingSide:  strings.Contains(castling, "k"),
		BlackQueenSide: strings.Contains(castling, "q"),
	}

	// Parse en passant square
	if parts[3] != "-" {
		sq, err := algebraicToSquare(parts[3])
		if err == nil {
			pos.EnPassantSquare = sq
		}
	}

	// Parse move counters if available
	if len(parts) >= 5 {
		fmt.Sscanf(parts[4], "%d", &pos.HalfMoveClock)
	}
	if len(parts) >= 6 {
		fmt.Sscanf(parts[5], "%d", &pos.FullMoveNumber)
	}

	return pos, nil
}

// ToFEN converts a ChessPosition back to FEN string
func (pos *ChessPosition) ToFEN() string {
	var fen strings.Builder

	// Piece placement
	pieceChars := []rune{'P', 'N', 'B', 'R', 'Q', 'K', 'p', 'n', 'b', 'r', 'q', 'k'}

	for rank := 0; rank < 8; rank++ {
		emptyCount := 0
		for file := 0; file < 8; file++ {
			pieceFound := false
			for ch := 0; ch < 12; ch++ {
				if pos.Board[ch][rank][file] > 0 {
					if emptyCount > 0 {
						fen.WriteRune(rune('0' + emptyCount))
						emptyCount = 0
					}
					fen.WriteRune(pieceChars[ch])
					pieceFound = true
					break
				}
			}
			if !pieceFound {
				emptyCount++
			}
		}
		if emptyCount > 0 {
			fen.WriteRune(rune('0' + emptyCount))
		}
		if rank < 7 {
			fen.WriteRune('/')
		}
	}

	// Active color
	if pos.WhiteToMove {
		fen.WriteString(" w ")
	} else {
		fen.WriteString(" b ")
	}

	// Castling rights
	castling := ""
	if pos.CastlingRights.WhiteKingSide {
		castling += "K"
	}
	if pos.CastlingRights.WhiteQueenSide {
		castling += "Q"
	}
	if pos.CastlingRights.BlackKingSide {
		castling += "k"
	}
	if pos.CastlingRights.BlackQueenSide {
		castling += "q"
	}
	if castling == "" {
		castling = "-"
	}
	fen.WriteString(castling + " ")

	// En passant
	if pos.EnPassantSquare >= 0 {
		fen.WriteString(squareToAlgebraic(pos.EnPassantSquare))
	} else {
		fen.WriteString("-")
	}

	// Move counters
	fen.WriteString(fmt.Sprintf(" %d %d", pos.HalfMoveClock, pos.FullMoveNumber))

	return fen.String()
}

// EvaluatePosition provides a simple material-based position evaluation
func (c *ChessAdapter) EvaluatePosition(board [12][8][8]float32) PositionEval {
	eval := PositionEval{
		Material: make(map[string]int),
	}

	// Piece values (centipawns)
	pieceValues := map[int]int{
		0:  100,  // White Pawn
		1:  320,  // White Knight
		2:  330,  // White Bishop
		3:  500,  // White Rook
		4:  900,  // White Queen
		5:  0,    // White King (not counted in material)
		6:  -100, // Black Pawn
		7:  -320, // Black Knight
		8:  -330, // Black Bishop
		9:  -500, // Black Rook
		10: -900, // Black Queen
		11: 0,    // Black King
	}

	pieceNames := []string{"P", "N", "B", "R", "Q", "K", "p", "n", "b", "r", "q", "k"}

	totalMaterial := 0
	pieceCount := make(map[int]int)

	for ch := 0; ch < 12; ch++ {
		count := 0
		for r := 0; r < 8; r++ {
			for f := 0; f < 8; f++ {
				if board[ch][r][f] > 0 {
					count++
					totalMaterial += pieceValues[ch]
				}
			}
		}
		pieceCount[ch] = count
		eval.Material[pieceNames[ch]] = count
	}

	eval.MaterialScore = totalMaterial

	// Calculate game phase (opening, middlegame, endgame)
	totalPieces := 0
	for ch := 0; ch < 12; ch++ {
		if ch != 5 && ch != 11 { // Exclude kings
			totalPieces += pieceCount[ch]
		}
	}

	if totalPieces >= 24 {
		eval.GamePhase = "opening"
	} else if totalPieces >= 12 {
		eval.GamePhase = "middlegame"
	} else {
		eval.GamePhase = "endgame"
	}

	// Detect special positions
	eval.IsEndgame = (totalPieces <= 10)
	eval.HasQueens = (pieceCount[4] > 0 || pieceCount[10] > 0)

	return eval
}

// PositionEval contains position evaluation details
type PositionEval struct {
	MaterialScore int            `json:"material_score"` // In centipawns
	Material      map[string]int `json:"material"`       // Piece counts
	GamePhase     string         `json:"game_phase"`     // "opening", "middlegame", "endgame"
	IsEndgame     bool           `json:"is_endgame"`
	HasQueens     bool           `json:"has_queens"`
}

// String returns human-readable evaluation
func (pe PositionEval) String() string {
	return fmt.Sprintf(
		"Eval{Score: %+d cp, Phase: %s, Queens: %v}",
		pe.MaterialScore,
		pe.GamePhase,
		pe.HasQueens,
	)
}

// IsMoveLegal performs basic move legality check
// Note: This is a simplified check. Full legal move validation requires chess engine integration
func (c *ChessAdapter) IsMoveLegal(board [12][8][8]float32, fromSquare, toSquare int) (bool, string) {
	if fromSquare < 0 || fromSquare >= 64 || toSquare < 0 || toSquare >= 64 {
		return false, "squares out of bounds"
	}

	fromRank := fromSquare / 8
	fromFile := fromSquare % 8
	toRank := toSquare / 8
	toFile := toSquare % 8

	// Check if there's a piece on the from square
	hasPiece := false
	movingPieceChannel := -1
	for ch := 0; ch < 12; ch++ {
		if board[ch][fromRank][fromFile] > 0 {
			hasPiece = true
			movingPieceChannel = ch
			break
		}
	}

	if !hasPiece {
		return false, "no piece on from square"
	}

	// Check if moving onto own piece
	isWhite := movingPieceChannel < 6
	for ch := 0; ch < 12; ch++ {
		if board[ch][toRank][toFile] > 0 {
			capturedIsWhite := ch < 6
			if isWhite == capturedIsWhite {
				return false, "cannot capture own piece"
			}
		}
	}

	// Basic movement pattern check
	rankDiff := abs(toRank - fromRank)
	fileDiff := abs(toFile - fromFile)

	pieceType := movingPieceChannel % 6 // 0=Pawn, 1=Knight, 2=Bishop, 3=Rook, 4=Queen, 5=King

	switch pieceType {
	case 0: // Pawn
		// Very simplified - doesn't check direction, en passant, etc.
		if fileDiff > 1 || rankDiff > 2 {
			return false, "invalid pawn move distance"
		}

	case 1: // Knight
		if !((rankDiff == 2 && fileDiff == 1) || (rankDiff == 1 && fileDiff == 2)) {
			return false, "invalid knight move"
		}

	case 2: // Bishop
		if rankDiff != fileDiff {
			return false, "bishop must move diagonally"
		}

	case 3: // Rook
		if rankDiff != 0 && fileDiff != 0 {
			return false, "rook must move along rank or file"
		}

	case 4: // Queen
		if rankDiff != fileDiff && rankDiff != 0 && fileDiff != 0 {
			return false, "invalid queen move"
		}

	case 5: // King
		if rankDiff > 1 || fileDiff > 1 {
			return false, "king can only move one square"
		}
	}

	return true, "ok"
}

// GetTopKMoves returns the top K moves from a prediction with legal move filtering
func (c *ChessAdapter) GetTopKMoves(pred tensor.Tensor, board [12][8][8]float32, k int) []MoveConfidence {
	data := pred.Data().([]float64)

	// Collect all moves with their probabilities
	type moveProb struct {
		index int
		prob  float64
		from  int
		to    int
	}

	moves := make([]moveProb, 0, 4096)
	for i := 0; i < 4096; i++ {
		if data[i] > 0.001 { // Filter out very low probabilities
			from := i / 64
			to := i % 64
			legal, _ := c.IsMoveLegal(board, from, to)
			if legal {
				moves = append(moves, moveProb{
					index: i,
					prob:  data[i],
					from:  from,
					to:    to,
				})
			}
		}
	}

	// Sort by probability (descending)
	for i := 0; i < len(moves); i++ {
		for j := i + 1; j < len(moves); j++ {
			if moves[j].prob > moves[i].prob {
				moves[i], moves[j] = moves[j], moves[i]
			}
		}
		if i >= k-1 {
			break
		}
	}

	// Extract top K
	topK := make([]MoveConfidence, 0, k)
	for i := 0; i < len(moves) && i < k; i++ {
		move := squareToAlgebraic(moves[i].from) + squareToAlgebraic(moves[i].to)
		topK = append(topK, MoveConfidence{
			Move:       move,
			FromSquare: moves[i].from,
			ToSquare:   moves[i].to,
			Confidence: moves[i].prob,
			Rank:       i + 1,
		})
	}

	return topK
}

// MoveConfidence represents a move with confidence score
type MoveConfidence struct {
	Move       string  `json:"move"`
	FromSquare int     `json:"from_square"`
	ToSquare   int     `json:"to_square"`
	Confidence float64 `json:"confidence"`
	Rank       int     `json:"rank"`
}

// String returns human-readable move confidence
func (mc MoveConfidence) String() string {
	return fmt.Sprintf("#%d %s (%.2f%%)", mc.Rank, mc.Move, mc.Confidence*100)
}

// GetSquareInfo returns information about a square
func (c *ChessAdapter) GetSquareInfo(board [12][8][8]float32, square int) SquareInfo {
	if square < 0 || square >= 64 {
		return SquareInfo{Square: square, IsEmpty: true}
	}

	rank := square / 8
	file := square % 8

	info := SquareInfo{
		Square:  square,
		Rank:    rank,
		File:    file,
		Algebra: squareToAlgebraic(square),
		IsEmpty: true,
	}

	// Check for piece
	pieceNames := []string{"P", "N", "B", "R", "Q", "K", "p", "n", "b", "r", "q", "k"}
	for ch := 0; ch < 12; ch++ {
		if board[ch][rank][file] > 0 {
			info.IsEmpty = false
			info.Piece = pieceNames[ch]
			info.PieceChannel = ch
			info.IsWhite = (ch < 6)
			break
		}
	}

	// Square color
	info.IsLightSquare = ((rank + file) % 2) == 0

	return info
}

// SquareInfo contains detailed information about a chess square
type SquareInfo struct {
	Square        int    `json:"square"`  // 0-63
	Rank          int    `json:"rank"`    // 0-7
	File          int    `json:"file"`    // 0-7
	Algebra       string `json:"algebra"` // "e4"
	IsEmpty       bool   `json:"is_empty"`
	Piece         string `json:"piece"`         // "P", "n", etc.
	PieceChannel  int    `json:"piece_channel"` // 0-11
	IsWhite       bool   `json:"is_white"`
	IsLightSquare bool   `json:"is_light_square"`
}

// Helper function
func abs(x int) int {
	if x < 0 {
		return -x
	}
	return x
}
