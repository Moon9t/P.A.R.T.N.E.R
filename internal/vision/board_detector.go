package vision

import (
	"errors"
	"fmt"
	"image"
	"math"

	"gocv.io/x/gocv"
)

// PieceType represents a chess piece type
type PieceType int

const (
	Empty PieceType = iota
	WhitePawn
	WhiteKnight
	WhiteBishop
	WhiteRook
	WhiteQueen
	WhiteKing
	BlackPawn
	BlackKnight
	BlackBishop
	BlackRook
	BlackQueen
	BlackKing
)

// BoardDetector detects chess piece positions on a board
type BoardDetector struct {
	squareSize      int
	colorThresholds ColorThresholds
	useGrayscale    bool
}

// ColorThresholds defines color ranges for piece detection
type ColorThresholds struct {
	// HSV ranges for detecting pieces
	WhiteLower gocv.Scalar // Lower bound for white pieces
	WhiteUpper gocv.Scalar // Upper bound for white pieces
	BlackLower gocv.Scalar // Lower bound for black pieces
	BlackUpper gocv.Scalar // Upper bound for black pieces
	EmptyLower gocv.Scalar // Lower bound for empty squares
	EmptyUpper gocv.Scalar // Upper bound for empty squares
}

// DefaultColorThresholds returns default color thresholds
func DefaultColorThresholds() ColorThresholds {
	return ColorThresholds{
		// White pieces (light colored in HSV)
		WhiteLower: gocv.NewScalar(0, 0, 180, 0),
		WhiteUpper: gocv.NewScalar(180, 50, 255, 0),
		// Black pieces (dark colored in HSV)
		BlackLower: gocv.NewScalar(0, 0, 0, 0),
		BlackUpper: gocv.NewScalar(180, 255, 80, 0),
		// Empty squares (medium brightness)
		EmptyLower: gocv.NewScalar(0, 0, 80, 0),
		EmptyUpper: gocv.NewScalar(180, 50, 180, 0),
	}
}

// NewBoardDetector creates a new board detector
func NewBoardDetector(squareSize int, useGrayscale bool) *BoardDetector {
	return &BoardDetector{
		squareSize:      squareSize,
		colorThresholds: DefaultColorThresholds(),
		useGrayscale:    useGrayscale,
	}
}

// DetectBoard extracts the board state from a captured frame
// Returns a 12x8x8 tensor representing the board state
func (bd *BoardDetector) DetectBoard(frame *gocv.Mat) ([12][8][8]float32, error) {
	if frame.Empty() {
		return [12][8][8]float32{}, errors.New("empty frame")
	}

	var tensor [12][8][8]float32

	// Resize frame to 8x8 grid of squares
	squareSize := bd.squareSize
	expectedSize := squareSize * 8

	resized := gocv.NewMat()
	defer resized.Close()
	gocv.Resize(*frame, &resized, image.Pt(expectedSize, expectedSize), 0, 0, gocv.InterpolationLinear)

	// Process each square
	for rank := 0; rank < 8; rank++ {
		for file := 0; file < 8; file++ {
			// Extract square region
			x := file * squareSize
			y := rank * squareSize
			squareRect := image.Rect(x, y, x+squareSize, y+squareSize)
			square := resized.Region(squareRect)

			// Detect piece in this square
			pieceType, confidence := bd.detectPieceInSquare(&square)
			square.Close()

			// Convert to tensor representation
			if pieceType != Empty && confidence > 0.5 {
				channel := bd.pieceTypeToChannel(pieceType)
				if channel >= 0 && channel < 12 {
					tensor[channel][rank][file] = 1.0
				}
			}
		}
	}

	return tensor, nil
}

// detectPieceInSquare detects what piece is in a square
// Returns the piece type and confidence (0-1)
func (bd *BoardDetector) detectPieceInSquare(square *gocv.Mat) (PieceType, float32) {
	if square.Empty() {
		return Empty, 0.0
	}

	if bd.useGrayscale {
		return bd.detectPieceGrayscale(square)
	}
	return bd.detectPieceColor(square)
}

// detectPieceGrayscale uses intensity-based detection
func (bd *BoardDetector) detectPieceGrayscale(square *gocv.Mat) (PieceType, float32) {
	// Convert to grayscale
	gray := gocv.NewMat()
	defer gray.Close()
	gocv.CvtColor(*square, &gray, gocv.ColorBGRToGray)

	// Calculate mean intensity
	mean := gray.Mean()
	intensity := mean.Val1

	// Simple threshold-based detection
	// Dark = black piece, Light = white piece, Medium = empty
	confidence := float32(0.7) // Default confidence

	if intensity < 80 {
		// Dark - likely black piece
		return BlackPawn, confidence // Default to pawn, would need more sophisticated detection
	} else if intensity > 180 {
		// Light - likely white piece
		return WhitePawn, confidence
	} else {
		// Medium - likely empty
		return Empty, confidence
	}
}

// detectPieceColor uses color-based detection with HSV
func (bd *BoardDetector) detectPieceColor(square *gocv.Mat) (PieceType, float32) {
	// Convert to HSV
	hsv := gocv.NewMat()
	defer hsv.Close()
	gocv.CvtColor(*square, &hsv, gocv.ColorBGRToHSV)

	// Create masks for each color range
	whiteMask := gocv.NewMat()
	defer whiteMask.Close()
	blackMask := gocv.NewMat()
	defer blackMask.Close()
	emptyMask := gocv.NewMat()
	defer emptyMask.Close()

	gocv.InRangeWithScalar(hsv, bd.colorThresholds.WhiteLower, bd.colorThresholds.WhiteUpper, &whiteMask)
	gocv.InRangeWithScalar(hsv, bd.colorThresholds.BlackLower, bd.colorThresholds.BlackUpper, &blackMask)
	gocv.InRangeWithScalar(hsv, bd.colorThresholds.EmptyLower, bd.colorThresholds.EmptyUpper, &emptyMask)

	// Count pixels in each mask
	whitePixels := gocv.CountNonZero(whiteMask)
	blackPixels := gocv.CountNonZero(blackMask)
	emptyPixels := gocv.CountNonZero(emptyMask)

	totalPixels := square.Rows() * square.Cols()
	whiteRatio := float32(whitePixels) / float32(totalPixels)
	blackRatio := float32(blackPixels) / float32(totalPixels)
	emptyRatio := float32(emptyPixels) / float32(totalPixels)

	// Determine piece type based on highest ratio
	maxRatio := float32(math.Max(float64(whiteRatio), math.Max(float64(blackRatio), float64(emptyRatio))))

	if maxRatio < 0.3 {
		return Empty, 0.0 // Low confidence
	}

	if whiteRatio == maxRatio {
		return WhitePawn, maxRatio
	} else if blackRatio == maxRatio {
		return BlackPawn, maxRatio
	}

	return Empty, maxRatio
}

// pieceTypeToChannel converts a piece type to tensor channel index
// Channels: 0-5 = white pieces, 6-11 = black pieces
// Order: Pawn, Knight, Bishop, Rook, Queen, King
func (bd *BoardDetector) pieceTypeToChannel(pt PieceType) int {
	switch pt {
	case WhitePawn:
		return 0
	case WhiteKnight:
		return 1
	case WhiteBishop:
		return 2
	case WhiteRook:
		return 3
	case WhiteQueen:
		return 4
	case WhiteKing:
		return 5
	case BlackPawn:
		return 6
	case BlackKnight:
		return 7
	case BlackBishop:
		return 8
	case BlackRook:
		return 9
	case BlackQueen:
		return 10
	case BlackKing:
		return 11
	default:
		return -1
	}
}

// DetectBoardDifference compares two board states and returns changed squares
func (bd *BoardDetector) DetectBoardDifference(board1, board2 [12][8][8]float32) []Position {
	var changes []Position

	for rank := 0; rank < 8; rank++ {
		for file := 0; file < 8; file++ {
			changed := false
			for channel := 0; channel < 12; channel++ {
				if board1[channel][rank][file] != board2[channel][rank][file] {
					changed = true
					break
				}
			}
			if changed {
				changes = append(changes, Position{Rank: rank, File: file})
			}
		}
	}

	return changes
}

// Position represents a square on the board
type Position struct {
	Rank int // 0-7
	File int // 0-7
}

// String returns algebraic notation (e.g., "e4")
func (p Position) String() string {
	files := "abcdefgh"
	return fmt.Sprintf("%c%d", files[p.File], p.Rank+1)
}

// ToSquareIndex converts position to 0-63 square index
func (p Position) ToSquareIndex() int {
	return p.Rank*8 + p.File
}

// BoardStateTensor represents a detected board state
type BoardStateTensor struct {
	Tensor    [12][8][8]float32
	Timestamp int64
	Changes   []Position
}

// ValidateBoardTensor checks if a board tensor is valid
func ValidateBoardTensor(tensor [12][8][8]float32) error {
	// Check for reasonable number of pieces
	totalPieces := 0
	for channel := 0; channel < 12; channel++ {
		for rank := 0; rank < 8; rank++ {
			for file := 0; file < 8; file++ {
				if tensor[channel][rank][file] > 0 {
					totalPieces++
				}
			}
		}
	}

	if totalPieces < 2 {
		return fmt.Errorf("too few pieces detected: %d", totalPieces)
	}
	if totalPieces > 32 {
		return fmt.Errorf("too many pieces detected: %d", totalPieces)
	}

	// Check for multiple pieces on same square
	for rank := 0; rank < 8; rank++ {
		for file := 0; file < 8; file++ {
			count := 0
			for channel := 0; channel < 12; channel++ {
				if tensor[channel][rank][file] > 0 {
					count++
				}
			}
			if count > 1 {
				return fmt.Errorf("multiple pieces on same square: rank=%d, file=%d, count=%d", rank, file, count)
			}
		}
	}

	return nil
}

// PrintBoardTensor prints a human-readable representation of the board
func PrintBoardTensor(tensor [12][8][8]float32) string {
	pieceSymbols := map[int]string{
		0:  "♙", // White Pawn
		1:  "♘", // White Knight
		2:  "♗", // White Bishop
		3:  "♖", // White Rook
		4:  "♕", // White Queen
		5:  "♔", // White King
		6:  "♟", // Black Pawn
		7:  "♞", // Black Knight
		8:  "♝", // Black Bishop
		9:  "♜", // Black Rook
		10: "♛", // Black Queen
		11: "♚", // Black King
	}

	result := "\n  a b c d e f g h\n"
	for rank := 7; rank >= 0; rank-- {
		result += fmt.Sprintf("%d ", rank+1)
		for file := 0; file < 8; file++ {
			found := false
			for channel := 0; channel < 12; channel++ {
				if tensor[channel][rank][file] > 0 {
					result += pieceSymbols[channel] + " "
					found = true
					break
				}
			}
			if !found {
				if (rank+file)%2 == 0 {
					result += "□ "
				} else {
					result += "■ "
				}
			}
		}
		result += fmt.Sprintf("%d\n", rank+1)
	}
	result += "  a b c d e f g h\n"

	return result
}
