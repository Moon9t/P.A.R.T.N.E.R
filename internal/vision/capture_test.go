package vision

import (
	"testing"
)

func TestNewCapturer(t *testing.T) {
	capturer := NewCapturer(0, 0, 640, 640, 8, 15.0)
	if capturer == nil {
		t.Fatal("Failed to create capturer")
	}

	if capturer.boardSize != 8 {
		t.Errorf("Expected board size 8, got %d", capturer.boardSize)
	}
}

func TestVisualizeBoard(t *testing.T) {
	grid := make([]float64, 64)
	for i := range grid {
		grid[i] = float64(i) / 64.0
	}

	visualization := VisualizeBoard(grid, 8)
	if visualization == "" {
		t.Error("Visualization should not be empty")
	}
}
