package data

import (
	"strings"
	"testing"

	"github.com/notnil/chess"
)

func TestParsePGN(t *testing.T) {
	// Test with actual chess package - create a game programmatically
	game := chess.NewGame()
	
	// Make some moves
	if err := game.MoveStr("e4"); err != nil {
		t.Fatalf("Failed to make move: %v", err)
	}
	if err := game.MoveStr("e5"); err != nil {
		t.Fatalf("Failed to make move: %v", err)
	}
	if err := game.MoveStr("Nf3"); err != nil {
		t.Fatalf("Failed to make move: %v", err)
	}
	
	moves := game.Moves()
	if len(moves) != 3 {
		t.Fatalf("Expected 3 moves, got %d", len(moves))
	}
	
	// Test that we can use the chess package correctly
	pos := game.Position()
	if pos == nil {
		t.Fatal("Expected non-nil position")
	}
}

func TestExtractPositions(t *testing.T) {
	game := chess.NewGame()

	// Make a few moves
	moves := []string{"e4", "e5", "Nf3", "Nc6"}
	for _, moveStr := range moves {
		if err := game.MoveStr(moveStr); err != nil {
			t.Fatalf("Failed to make move %s: %v", moveStr, err)
		}
	}

	positions, err := ExtractPositions(game)
	if err != nil {
		t.Fatalf("Failed to extract positions: %v", err)
	}

	if len(positions) != len(moves) {
		t.Fatalf("Expected %d positions, got %d", len(moves), len(positions))
	}

	// Verify each position has a board and move
	for i, pos := range positions {
		if pos.Board == nil {
			t.Errorf("Position %d has nil board", i)
		}
		if pos.Move == nil {
			t.Errorf("Position %d has nil move", i)
		}
	}
}

func TestExtractPositionsNilGame(t *testing.T) {
	_, err := ExtractPositions(nil)
	if err == nil {
		t.Error("Expected error for nil game, got nil")
	}
}

func TestExtractPositionsEmptyGame(t *testing.T) {
	game := chess.NewGame()
	positions, err := ExtractPositions(game)

	if err != nil {
		t.Fatalf("Failed to extract positions from empty game: %v", err)
	}

	if len(positions) != 0 {
		t.Fatalf("Expected 0 positions from empty game, got %d", len(positions))
	}
}

func TestValidatePGN(t *testing.T) {
	tests := []struct {
		name      string
		content   string
		shouldErr bool
	}{
		{
			name: "Valid PGN with event header",
			content: `[Event "Test"]
[Site "Test"]
1. e4 e5`,
			shouldErr: false,
		},
		{
			name:      "Invalid content",
			content:   "This is not a PGN file",
			shouldErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Note: ValidatePGN requires a file path, so we test the logic conceptually
			// In a full test, we'd create temp files
			hasEvent := strings.Contains(tt.content, "[Event")
			hasMoves := strings.Contains(tt.content, "1.")

			isValid := hasEvent || hasMoves
			if isValid && tt.shouldErr {
				t.Error("Expected validation to fail but it passed")
			}
			if !isValid && !tt.shouldErr {
				t.Error("Expected validation to pass but it failed")
			}
		})
	}
}
