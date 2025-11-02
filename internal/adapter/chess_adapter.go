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
