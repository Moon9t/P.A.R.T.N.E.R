package data

import (
	"testing"

	"github.com/notnil/chess"
)

func TestPieceToChannel(t *testing.T) {
	tests := []struct {
		piece           chess.Piece
		expectedChannel int
	}{
		{chess.WhitePawn, 0},
		{chess.WhiteKnight, 1},
		{chess.WhiteBishop, 2},
		{chess.WhiteRook, 3},
		{chess.WhiteQueen, 4},
		{chess.WhiteKing, 5},
		{chess.BlackPawn, 6},
		{chess.BlackKnight, 7},
		{chess.BlackBishop, 8},
		{chess.BlackRook, 9},
		{chess.BlackQueen, 10},
		{chess.BlackKing, 11},
	}

	for _, tt := range tests {
		channel := PieceToChannel(tt.piece)
		if channel != tt.expectedChannel {
			t.Errorf("PieceToChannel(%v) = %d, want %d", tt.piece, channel, tt.expectedChannel)
		}
	}
}

func TestTensorizeBoard(t *testing.T) {
	// Create a new game with starting position
	game := chess.NewGame()
	board := game.Position().Board()

	tensor, err := TensorizeBoard(board)
	if err != nil {
		t.Fatalf("Failed to tensorize board: %v", err)
	}

	// Verify tensor shape
	if len(tensor) != NumChannels {
		t.Errorf("Expected %d channels, got %d", NumChannels, len(tensor))
	}

	for i := range tensor {
		if len(tensor[i]) != BoardSize {
			t.Errorf("Channel %d: expected %d rows, got %d", i, BoardSize, len(tensor[i]))
		}
		for j := range tensor[i] {
			if len(tensor[i][j]) != BoardSize {
				t.Errorf("Channel %d, Row %d: expected %d columns, got %d", i, j, BoardSize, len(tensor[i][j]))
			}
		}
	}

	// Check that some pieces are present
	// White pawns should be in channel 0, rank 6 (second from bottom in our representation)
	whitePawnCount := 0
	for file := 0; file < BoardSize; file++ {
		if tensor[0][6][file] == 1.0 {
			whitePawnCount++
		}
	}
	if whitePawnCount != 8 {
		t.Errorf("Expected 8 white pawns, found %d", whitePawnCount)
	}

	// Black pawns should be in channel 6, rank 1
	blackPawnCount := 0
	for file := 0; file < BoardSize; file++ {
		if tensor[6][1][file] == 1.0 {
			blackPawnCount++
		}
	}
	if blackPawnCount != 8 {
		t.Errorf("Expected 8 black pawns, found %d", blackPawnCount)
	}
}

func TestTensorizeBoardNil(t *testing.T) {
	_, err := TensorizeBoard(nil)
	if err == nil {
		t.Error("Expected error for nil board, got nil")
	}
}

func TestEncodeMoveLabel(t *testing.T) {
	// Create a move: e2 to e4
	game := chess.NewGame()
	actualMove := game.ValidMoves()[0] // Get a valid move

	fromSquare, toSquare, err := EncodeMoveLabel(actualMove)
	if err != nil {
		t.Fatalf("Failed to encode move: %v", err)
	}

	if fromSquare < 0 || fromSquare >= 64 {
		t.Errorf("Invalid from square: %d", fromSquare)
	}

	if toSquare < 0 || toSquare >= 64 {
		t.Errorf("Invalid to square: %d", toSquare)
	}
}

func TestEncodeMoveLabel_Nil(t *testing.T) {
	_, _, err := EncodeMoveLabel(nil)
	if err == nil {
		t.Error("Expected error for nil move, got nil")
	}
}

func TestDecodeMoveLabel(t *testing.T) {
	tests := []struct {
		name        string
		fromSquare  int
		toSquare    int
		expectError bool
	}{
		{"Valid move", 12, 28, false},
		{"Edge case - corner to corner", 0, 63, false},
		{"Invalid from square", -1, 28, true},
		{"Invalid to square", 12, 64, true},
		{"Both invalid", -1, 64, true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, _, err := DecodeMoveLabel(tt.fromSquare, tt.toSquare)

			if tt.expectError && err == nil {
				t.Error("Expected error but got nil")
			}
			if !tt.expectError && err != nil {
				t.Errorf("Unexpected error: %v", err)
			}
		})
	}
}

func TestValidateTensor(t *testing.T) {
	t.Run("Valid tensor", func(t *testing.T) {
		var tensor [NumChannels][BoardSize][BoardSize]float32

		// Place some pieces
		tensor[0][0][0] = 1.0 // White pawn at a1
		tensor[6][7][7] = 1.0 // Black pawn at h8

		err := ValidateTensor(tensor)
		if err != nil {
			t.Errorf("Valid tensor failed validation: %v", err)
		}
	})

	t.Run("Invalid value", func(t *testing.T) {
		var tensor [NumChannels][BoardSize][BoardSize]float32
		tensor[0][0][0] = 0.5 // Invalid value

		err := ValidateTensor(tensor)
		if err == nil {
			t.Error("Expected error for invalid tensor value")
		}
	})

	t.Run("Multiple pieces on same square", func(t *testing.T) {
		var tensor [NumChannels][BoardSize][BoardSize]float32
		tensor[0][0][0] = 1.0 // White pawn
		tensor[1][0][0] = 1.0 // White knight (same square!)

		err := ValidateTensor(tensor)
		if err == nil {
			t.Error("Expected error for multiple pieces on same square")
		}
	})
}

func TestTensorToFlatArray(t *testing.T) {
	var tensor [NumChannels][BoardSize][BoardSize]float32

	// Set a few values
	tensor[0][0][0] = 1.0
	tensor[5][3][4] = 1.0
	tensor[11][7][7] = 1.0

	flat := TensorToFlatArray(tensor)

	expectedLen := NumChannels * BoardSize * BoardSize
	if len(flat) != expectedLen {
		t.Errorf("Expected flat array length %d, got %d", expectedLen, len(flat))
	}

	// Check that the values we set are present
	nonZeroCount := 0
	for _, v := range flat {
		if v != 0.0 {
			nonZeroCount++
		}
	}

	if nonZeroCount != 3 {
		t.Errorf("Expected 3 non-zero values, got %d", nonZeroCount)
	}
}

func TestFlatArrayToTensor(t *testing.T) {
	t.Run("Valid conversion", func(t *testing.T) {
		expectedLen := NumChannels * BoardSize * BoardSize
		flat := make([]float32, expectedLen)

		// Set some values
		flat[0] = 1.0
		flat[100] = 1.0
		flat[expectedLen-1] = 1.0

		tensor, err := FlatArrayToTensor(flat)
		if err != nil {
			t.Fatalf("Failed to convert flat array to tensor: %v", err)
		}

		// Convert back and compare
		flat2 := TensorToFlatArray(tensor)
		if len(flat) != len(flat2) {
			t.Errorf("Length mismatch after round-trip conversion")
		}

		for i := range flat {
			if flat[i] != flat2[i] {
				t.Errorf("Value mismatch at index %d: %f != %f", i, flat[i], flat2[i])
			}
		}
	})

	t.Run("Invalid length", func(t *testing.T) {
		flat := make([]float32, 100) // Wrong size

		_, err := FlatArrayToTensor(flat)
		if err == nil {
			t.Error("Expected error for invalid flat array length")
		}
	})
}

func TestRoundTripTensorConversion(t *testing.T) {
	// Create a game and tensorize it
	game := chess.NewGame()
	board := game.Position().Board()

	tensor1, err := TensorizeBoard(board)
	if err != nil {
		t.Fatalf("Failed to tensorize board: %v", err)
	}

	// Convert to flat array
	flat := TensorToFlatArray(tensor1)

	// Convert back to tensor
	tensor2, err := FlatArrayToTensor(flat)
	if err != nil {
		t.Fatalf("Failed to convert flat array to tensor: %v", err)
	}

	// Compare tensors
	for c := 0; c < NumChannels; c++ {
		for r := 0; r < BoardSize; r++ {
			for f := 0; f < BoardSize; f++ {
				if tensor1[c][r][f] != tensor2[c][r][f] {
					t.Errorf("Mismatch at [%d][%d][%d]: %f != %f",
						c, r, f, tensor1[c][r][f], tensor2[c][r][f])
				}
			}
		}
	}
}
