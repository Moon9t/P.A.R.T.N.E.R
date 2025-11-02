package iface

import (
	"testing"

	"github.com/thyrook/partner/internal/config"
	"github.com/thyrook/partner/internal/model"
)

func TestCLICreation(t *testing.T) {
	cfg := config.DefaultConfig()
	cli := NewCLI(cfg, false)

	if cli == nil {
		t.Fatal("Failed to create CLI")
	}

	if cli.config != cfg {
		t.Error("Config not set correctly")
	}
}

func TestPrintMove(t *testing.T) {
	cfg := config.DefaultConfig()
	cli := NewCLI(cfg, true) // Quiet mode for testing

	move := model.MoveScore{
		MoveIndex: 260, // e2e4
		Score:     0.75,
	}

	// Should not panic
	cli.PrintMove(move)
}

func TestPrintTopMoves(t *testing.T) {
	cfg := config.DefaultConfig()
	cli := NewCLI(cfg, true)

	moves := []model.MoveScore{
		{MoveIndex: 260, Score: 0.75},
		{MoveIndex: 261, Score: 0.60},
		{MoveIndex: 262, Score: 0.45},
	}

	// Should not panic
	cli.PrintTopMoves(moves, 3)
}

func TestPrintStatus(t *testing.T) {
	cfg := config.DefaultConfig()
	cli := NewCLI(cfg, true)

	levels := []string{"info", "success", "warning", "error"}
	for _, level := range levels {
		cli.PrintStatus("Test message", level)
	}
}

func TestPrintProgress(t *testing.T) {
	cfg := config.DefaultConfig()
	cli := NewCLI(cfg, true)

	cli.PrintProgress(50, 100, "Testing")
	cli.PrintProgress(100, 100, "Complete")
}

func TestPrintAnalysisResults(t *testing.T) {
	cfg := config.DefaultConfig()
	cli := NewCLI(cfg, true)

	topKAccuracy := map[int]float64{
		3:  65.5,
		5:  75.2,
		10: 85.0,
	}

	cli.PrintAnalysisResults(1000, 450, topKAccuracy)
}

func TestMoveToNaturalLanguage(t *testing.T) {
	cfg := config.DefaultConfig()
	cli := NewCLI(cfg, true)

	tests := []struct {
		notation string
		expected string
	}{
		{"e2e4", "Move pawn to e4"},
		{"g1f3", "Move knight to f3"},
		{"f1c4", "Move bishop to c4"},
	}

	for _, tt := range tests {
		result := cli.moveToNaturalLanguage(tt.notation)
		if result != tt.expected {
			t.Errorf("moveToNaturalLanguage(%s) = %s, want %s", tt.notation, result, tt.expected)
		}
	}
}

func TestGetRankEmoji(t *testing.T) {
	cfg := config.DefaultConfig()
	cli := NewCLI(cfg, true)

	tests := []struct {
		rank     int
		expected string
	}{
		{1, "🥇"},
		{2, "🥈"},
		{3, "🥉"},
		{4, "4."},
	}

	for _, tt := range tests {
		result := cli.getRankEmoji(tt.rank)
		if result != tt.expected {
			t.Errorf("getRankEmoji(%d) = %s, want %s", tt.rank, result, tt.expected)
		}
	}
}

func TestQuietMode(t *testing.T) {
	cfg := config.DefaultConfig()

	// Test quiet mode suppresses output
	cli := NewCLI(cfg, true)
	if !cli.quiet {
		t.Error("Quiet mode not set")
	}

	// Should not panic
	cli.PrintBanner()
	cli.PrintModeHeader("test")
	cli.PrintStatus("test", "info")
	cli.PrintSeparator()
}

func TestPrintTrainingStats(t *testing.T) {
	cfg := config.DefaultConfig()
	cli := NewCLI(cfg, true)

	// Should not panic
	cli.PrintTrainingStats(5, 10, 0.234, 45.6, 1500)
}
