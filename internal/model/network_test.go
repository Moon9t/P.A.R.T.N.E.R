package model

import (
	"testing"
)

func TestDecodeMove(t *testing.T) {
	tests := []struct {
		index    int
		expected string
	}{
		{0, "a1a1"},    // from=0 (a1), to=0 (a1) → 0*64+0=0
		{7, "a1h1"},    // from=0 (a1), to=7 (h1) → 0*64+7=7
		{63, "a1h8"},   // from=0 (a1), to=63 (h8) → 0*64+63=63
		{4032, "h8a1"}, // from=63 (h8), to=0 (a1) → 63*64+0=4032
		{4095, "h8h8"}, // from=63 (h8), to=63 (h8) → 63*64+63=4095
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
		{"a1a1", 0},    // from=a1(0), to=a1(0) → 0*64+0=0
		{"a1h1", 7},    // from=a1(0), to=h1(7) → 0*64+7=7
		{"a1h8", 63},   // from=a1(0), to=h8(63) → 0*64+63=63
		{"h8a1", 4032}, // from=h8(63), to=a1(0) → 63*64+0=4032
		{"h8h8", 4095}, // from=h8(63), to=h8(63) → 63*64+63=4095
		{"e2e4", 796},  // from=e2(12), to=e4(28) → 12*64+28=796
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
