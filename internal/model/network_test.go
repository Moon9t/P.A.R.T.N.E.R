package model

import (
	"testing"
)

func TestDecodeMove(t *testing.T) {
	tests := []struct {
		index    int
		expected string
	}{
		{0, "a1a1"},
		{63, "a1h1"},
		{4032, "h8a8"},
		{4095, "h8h8"},
	}

	for _, tt := range tests {
		result := DecodeMove(tt.index)
		if result != tt.expected {
			t.Errorf("DecodeMove(%d) = %s; want %s", tt.index, result, tt.expected)
		}
	}
}

func TestEncodeMove(t *testing.T) {
	tests := []struct {
		move     string
		expected int
	}{
		{"a1a1", 0},
		{"a1h1", 63},
		{"h8a8", 4032},
		{"h8h8", 4095},
		{"e2e4", 796},
	}

	for _, tt := range tests {
		result, err := EncodeMove(tt.move)
		if err != nil {
			t.Errorf("EncodeMove(%s) returned error: %v", tt.move, err)
			continue
		}
		if result != tt.expected {
			t.Errorf("EncodeMove(%s) = %d; want %d", tt.move, result, tt.expected)
		}
	}
}

func TestGetTopKMoves(t *testing.T) {
	predictions := make([]float64, 10)
	for i := range predictions {
		predictions[i] = float64(i) / 10.0
	}

	topMoves := GetTopKMoves(predictions, 3)

	if len(topMoves) != 3 {
		t.Errorf("Expected 3 top moves, got %d", len(topMoves))
	}

	// Check that they're in descending order
	for i := 1; i < len(topMoves); i++ {
		if topMoves[i].Score > topMoves[i-1].Score {
			t.Error("Top moves should be in descending order")
		}
	}
}
