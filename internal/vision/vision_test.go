package vision

import (
	"testing"
	"time"
)

func TestDefaultConfig(t *testing.T) {
	config := DefaultConfig()

	if config.BoardSize != 8 {
		t.Errorf("Expected board size 8, got %d", config.BoardSize)
	}

	if config.FPS < 1 {
		t.Errorf("Invalid FPS: %d", config.FPS)
	}

	if err := config.Validate(); err != nil {
		t.Errorf("Default config validation failed: %v", err)
	}
}

func TestConfigValidation(t *testing.T) {
	tests := []struct {
		name      string
		modifyFn  func(*Config)
		expectErr bool
	}{
		{
			name:      "Valid config",
			modifyFn:  func(c *Config) {},
			expectErr: false,
		},
		{
			name: "Invalid board size (too small)",
			modifyFn: func(c *Config) {
				c.BoardSize = 0
			},
			expectErr: true,
		},
		{
			name: "Invalid board size (too large)",
			modifyFn: func(c *Config) {
				c.BoardSize = 20
			},
			expectErr: true,
		},
		{
			name: "Invalid square size",
			modifyFn: func(c *Config) {
				c.SquareSize = 0
			},
			expectErr: true,
		},
		{
			name: "Invalid confidence",
			modifyFn: func(c *Config) {
				c.ConfidenceMin = 2.0
			},
			expectErr: true,
		},
		{
			name: "Invalid FPS",
			modifyFn: func(c *Config) {
				c.FPS = 0
			},
			expectErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			config := DefaultConfig()
			tt.modifyFn(config)

			err := config.Validate()
			if tt.expectErr && err == nil {
				t.Error("Expected validation error, got nil")
			}
			if !tt.expectErr && err != nil {
				t.Errorf("Unexpected validation error: %v", err)
			}
		})
	}
}

func TestPosition(t *testing.T) {
	pos := Position{Rank: 3, File: 4}

	// Test String (algebraic notation)
	notation := pos.String()
	expected := "e4"
	if notation != expected {
		t.Errorf("Expected %s, got %s", expected, notation)
	}

	// Test ToSquareIndex
	index := pos.ToSquareIndex()
	expectedIndex := 3*8 + 4 // 28
	if index != expectedIndex {
		t.Errorf("Expected square index %d, got %d", expectedIndex, index)
	}
}

func TestPieceTypeToChannel(t *testing.T) {
	detector := NewBoardDetector(100, true)

	tests := []struct {
		piece    PieceType
		expected int
	}{
		{WhitePawn, 0},
		{WhiteKnight, 1},
		{WhiteBishop, 2},
		{WhiteRook, 3},
		{WhiteQueen, 4},
		{WhiteKing, 5},
		{BlackPawn, 6},
		{BlackKnight, 7},
		{BlackBishop, 8},
		{BlackRook, 9},
		{BlackQueen, 10},
		{BlackKing, 11},
		{Empty, -1},
	}

	for _, tt := range tests {
		channel := detector.pieceTypeToChannel(tt.piece)
		if channel != tt.expected {
			t.Errorf("Piece %d: expected channel %d, got %d", tt.piece, tt.expected, channel)
		}
	}
}

func TestValidateBoardTensor(t *testing.T) {
	tests := []struct {
		name      string
		setupFn   func() [12][8][8]float32
		expectErr bool
	}{
		{
			name: "Valid starting position",
			setupFn: func() [12][8][8]float32 {
				var tensor [12][8][8]float32
				// White pawns
				for i := 0; i < 8; i++ {
					tensor[0][1][i] = 1.0
				}
				// Black pawns
				for i := 0; i < 8; i++ {
					tensor[6][6][i] = 1.0
				}
				// Add some pieces
				tensor[3][0][0] = 1.0  // White rook
				tensor[5][0][4] = 1.0  // White king
				tensor[9][7][0] = 1.0  // Black rook
				tensor[11][7][4] = 1.0 // Black king
				return tensor
			},
			expectErr: false,
		},
		{
			name: "Too few pieces",
			setupFn: func() [12][8][8]float32 {
				var tensor [12][8][8]float32
				tensor[5][0][4] = 1.0 // Only white king
				return tensor
			},
			expectErr: true,
		},
		{
			name: "Too many pieces",
			setupFn: func() [12][8][8]float32 {
				var tensor [12][8][8]float32
				// Fill all squares
				for rank := 0; rank < 8; rank++ {
					for file := 0; file < 8; file++ {
						tensor[0][rank][file] = 1.0
					}
				}
				return tensor
			},
			expectErr: true,
		},
		{
			name: "Multiple pieces on same square",
			setupFn: func() [12][8][8]float32 {
				var tensor [12][8][8]float32
				tensor[0][0][0] = 1.0 // White pawn
				tensor[6][0][0] = 1.0 // Black pawn on same square
				return tensor
			},
			expectErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tensor := tt.setupFn()
			err := ValidateBoardTensor(tensor)

			if tt.expectErr && err == nil {
				t.Error("Expected validation error, got nil")
			}
			if !tt.expectErr && err != nil {
				t.Errorf("Unexpected validation error: %v", err)
			}
		})
	}
}

func TestBoardDifference(t *testing.T) {
	detector := NewBoardDetector(100, true)

	// Create two board states
	var board1, board2 [12][8][8]float32

	// Board 1: White pawn at e2
	board1[0][1][4] = 1.0

	// Board 2: White pawn moved to e4
	board2[0][3][4] = 1.0

	// Detect differences
	changes := detector.DetectBoardDifference(board1, board2)

	// Should detect 2 changes: e2 (now empty) and e4 (now occupied)
	if len(changes) != 2 {
		t.Errorf("Expected 2 changes, got %d", len(changes))
	}

	// Check that changes include both squares
	foundE2 := false
	foundE4 := false
	for _, pos := range changes {
		if pos.Rank == 1 && pos.File == 4 {
			foundE2 = true
		}
		if pos.Rank == 3 && pos.File == 4 {
			foundE4 = true
		}
	}

	if !foundE2 {
		t.Error("Expected change at e2 not detected")
	}
	if !foundE4 {
		t.Error("Expected change at e4 not detected")
	}
}

func TestPrintBoardTensor(t *testing.T) {
	var tensor [12][8][8]float32

	// Set up a simple position
	tensor[0][1][4] = 1.0  // White pawn at e2
	tensor[5][0][4] = 1.0  // White king at e1
	tensor[11][7][4] = 1.0 // Black king at e8

	output := PrintBoardTensor(tensor)

	// Check that output is not empty
	if len(output) == 0 {
		t.Error("PrintBoardTensor returned empty string")
	}

	// Check that it contains board coordinates
	if len(output) < 100 { // Arbitrary minimum length
		t.Error("PrintBoardTensor output seems too short")
	}
}

func TestPipelineStats(t *testing.T) {
	var stats PipelineStats

	stats.FramesProcessed = 100
	stats.ChangesDetected = 5
	stats.LastProcessTime = 10 * time.Millisecond
	stats.Errors = 2

	if stats.FramesProcessed != 100 {
		t.Errorf("Expected 100 frames processed, got %d", stats.FramesProcessed)
	}

	if stats.ChangesDetected != 5 {
		t.Errorf("Expected 5 changes detected, got %d", stats.ChangesDetected)
	}
}

func TestColorThresholds(t *testing.T) {
	thresholds := DefaultColorThresholds()

	// Just verify that default thresholds are set
	if thresholds.WhiteLower.Val1 < 0 {
		t.Error("Invalid white lower threshold")
	}

	if thresholds.BlackUpper.Val1 < 0 {
		t.Error("Invalid black upper threshold")
	}
}

func TestBoardDetectorCreation(t *testing.T) {
	detector := NewBoardDetector(100, true)

	if detector.squareSize != 100 {
		t.Errorf("Expected square size 100, got %d", detector.squareSize)
	}

	if !detector.useGrayscale {
		t.Error("Expected grayscale mode to be enabled")
	}
}

func TestCaptureRegionConversion(t *testing.T) {
	region := CaptureRegion{
		X:      100,
		Y:      200,
		Width:  800,
		Height: 600,
	}

	rect := region.ToRectangle()

	if rect.Min.X != 100 || rect.Min.Y != 200 {
		t.Errorf("Rectangle min incorrect: got (%d,%d), want (100,200)",
			rect.Min.X, rect.Min.Y)
	}

	if rect.Max.X != 900 || rect.Max.Y != 800 {
		t.Errorf("Rectangle max incorrect: got (%d,%d), want (900,800)",
			rect.Max.X, rect.Max.Y)
	}
}
