package data

import (
	"fmt"

	"github.com/notnil/chess"
)

// TensorShape defines the shape of the board tensor: [12][8][8]
const (
	NumChannels = 12 // 6 piece types × 2 colors
	BoardSize   = 8
)

// PieceToChannel maps piece types to channel indices
// Channels 0-5: White pieces (Pawn, Knight, Bishop, Rook, Queen, King)
// Channels 6-11: Black pieces (Pawn, Knight, Bishop, Rook, Queen, King)
func PieceToChannel(piece chess.Piece) int {
	pieceType := piece.Type()
	color := piece.Color()

	var baseChannel int
	switch pieceType {
	case chess.Pawn:
		baseChannel = 0
	case chess.Knight:
		baseChannel = 1
	case chess.Bishop:
		baseChannel = 2
	case chess.Rook:
		baseChannel = 3
	case chess.Queen:
		baseChannel = 4
	case chess.King:
		baseChannel = 5
	default:
		return -1
	}

	if color == chess.Black {
		baseChannel += 6
	}

	return baseChannel
}

// TensorizeBoard converts a chess board to a [12][8][8] float32 tensor
// Each channel represents one piece type and color
func TensorizeBoard(board *chess.Board) ([NumChannels][BoardSize][BoardSize]float32, error) {
	var tensor [NumChannels][BoardSize][BoardSize]float32

	if board == nil {
		return tensor, fmt.Errorf("board is nil")
	}

	// Iterate through all squares
	for rank := 0; rank < 8; rank++ {
		for file := 0; file < 8; file++ {
			square := chess.Square((7-rank)*8 + file) // Convert to chess square index
			piece := board.Piece(square)

			if piece != chess.NoPiece {
				channel := PieceToChannel(piece)
				if channel >= 0 && channel < NumChannels {
					tensor[channel][rank][file] = 1.0
				}
			}
		}
	}

	return tensor, nil
}

// EncodeMoveLabel converts a chess move to a pair of square indices (from, to)
// Returns (fromSquare, toSquare) where each is in range [0, 63]
func EncodeMoveLabel(move *chess.Move) (int, int, error) {
	if move == nil {
		return 0, 0, fmt.Errorf("move is nil")
	}

	fromSquare := int(move.S1())
	toSquare := int(move.S2())

	if fromSquare < 0 || fromSquare >= 64 {
		return 0, 0, fmt.Errorf("invalid from square: %d", fromSquare)
	}

	if toSquare < 0 || toSquare >= 64 {
		return 0, 0, fmt.Errorf("invalid to square: %d", toSquare)
	}

	return fromSquare, toSquare, nil
}

// DecodeMoveLabel converts square indices back to a move representation
func DecodeMoveLabel(fromSquare, toSquare int) (chess.Square, chess.Square, error) {
	if fromSquare < 0 || fromSquare >= 64 {
		return chess.NoSquare, chess.NoSquare, fmt.Errorf("invalid from square: %d", fromSquare)
	}

	if toSquare < 0 || toSquare >= 64 {
		return chess.NoSquare, chess.NoSquare, fmt.Errorf("invalid to square: %d", toSquare)
	}

	return chess.Square(fromSquare), chess.Square(toSquare), nil
}

// ValidateTensor checks if a tensor has the correct shape and valid values
func ValidateTensor(tensor [NumChannels][BoardSize][BoardSize]float32) error {
	// Check that each square has at most one piece
	for rank := 0; rank < BoardSize; rank++ {
		for file := 0; file < BoardSize; file++ {
			pieceCount := 0
			for channel := 0; channel < NumChannels; channel++ {
				if tensor[channel][rank][file] != 0.0 {
					if tensor[channel][rank][file] != 1.0 {
						return fmt.Errorf("invalid tensor value at [%d][%d][%d]: expected 0.0 or 1.0, got %f",
							channel, rank, file, tensor[channel][rank][file])
					}
					pieceCount++
				}
			}
			if pieceCount > 1 {
				return fmt.Errorf("multiple pieces at square [%d][%d]", rank, file)
			}
		}
	}

	return nil
}

// TensorToFlatArray converts a 3D tensor to a flat float32 array
func TensorToFlatArray(tensor [NumChannels][BoardSize][BoardSize]float32) []float32 {
	flat := make([]float32, NumChannels*BoardSize*BoardSize)
	idx := 0
	for c := 0; c < NumChannels; c++ {
		for r := 0; r < BoardSize; r++ {
			for f := 0; f < BoardSize; f++ {
				flat[idx] = tensor[c][r][f]
				idx++
			}
		}
	}
	return flat
}

// FlatArrayToTensor converts a flat array back to a 3D tensor
func FlatArrayToTensor(flat []float32) ([NumChannels][BoardSize][BoardSize]float32, error) {
	var tensor [NumChannels][BoardSize][BoardSize]float32

	expectedLen := NumChannels * BoardSize * BoardSize
	if len(flat) != expectedLen {
		return tensor, fmt.Errorf("invalid flat array length: expected %d, got %d", expectedLen, len(flat))
	}

	idx := 0
	for c := 0; c < NumChannels; c++ {
		for r := 0; r < BoardSize; r++ {
			for f := 0; f < BoardSize; f++ {
				tensor[c][r][f] = flat[idx]
				idx++
			}
		}
	}

	return tensor, nil
}
