package iface

import (
	"fmt"
	"os"
	"strings"
	"time"

	"github.com/thyrook/partner/internal/config"
	"github.com/thyrook/partner/internal/model"
)

// CLI provides command-line interface utilities
type CLI struct {
	config *config.Config
	quiet  bool
}

// NewCLI creates a new CLI interface
func NewCLI(cfg *config.Config, quiet bool) *CLI {
	return &CLI{
		config: cfg,
		quiet:  quiet,
	}
}

// PrintBanner displays the application banner
func (c *CLI) PrintBanner() {
	if c.quiet {
		return
	}

	banner := `
╔══════════════════════════════════════════════════════════════════════╗
║                                                                      ║
║   ██████╗    █████╗   ██████╗  ████████╗ ███╗   ██╗ ███████╗ ██████╗  ║
║   ██╔══██╗  ██╔══██╗  ██╔══██╗ ╚══██╔══╝ ████╗  ██║ ██╔════╝ ██╔══██╗ ║
║   ██████╔╝  ███████║  ██████╔╝    ██║    ██╔██╗ ██║ █████╗   ██████╔╝ ║
║   ██╔═══╝   ██╔══██║  ██╔══██╗    ██║    ██║╚██╗██║ ██╔══╝   ██╔══██╗ ║
║   ██║       ██║  ██║  ██║  ██║    ██║    ██║ ╚████║ ███████╗ ██║  ██║ ║
║   ╚═╝       ╚═╝  ╚═╝  ╚═╝  ╚═╝    ╚═╝    ╚═╝  ╚═══╝ ╚══════╝ ╚═╝  ╚═╝ ║
║                                                                      ║
║        Predictive Analysis & Real-Time Neural Engine for Chess      ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
`
	fmt.Println(banner)
}

// PrintModeHeader displays the mode-specific header
func (c *CLI) PrintModeHeader(mode string) {
	if c.quiet {
		return
	}

	var header string
	switch mode {
	case "observe":
		header = `
🔍 OBSERVE MODE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Watching chess board for moves and providing real-time predictions.
Press Ctrl+C to stop.
`
	case "train":
		header = `
🎓 TRAIN MODE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Training model with dataset. This may take several minutes.
`
	case "analyze":
		header = `
📊 ANALYZE MODE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Running accuracy analysis on test dataset.
`
	default:
		header = fmt.Sprintf("\n%s MODE\n", strings.ToUpper(mode))
	}

	fmt.Println(header)
}

// PrintMove prints a move prediction in natural language
func (c *CLI) PrintMove(move model.MoveScore) {
	notation := model.DecodeMove(move.MoveIndex)
	naturalLang := c.moveToNaturalLanguage(notation)

	confidence := move.Score * 100

	if c.quiet {
		fmt.Printf("%s (%.1f%%)\n", notation, confidence)
	} else {
		fmt.Printf("🎯 %s\n", naturalLang)
		fmt.Printf("   Notation: %s | Confidence: %.1f%%\n", notation, confidence)
	}
}

// PrintMoveWithRank prints a move with its rank
func (c *CLI) PrintMoveWithRank(rank int, move model.MoveScore) {
	notation := model.DecodeMove(move.MoveIndex)
	naturalLang := c.moveToNaturalLanguage(notation)
	confidence := move.Score * 100

	if c.quiet {
		fmt.Printf("%d. %s (%.1f%%)\n", rank, notation, confidence)
	} else {
		emoji := c.getRankEmoji(rank)
		fmt.Printf("%s %s\n", emoji, naturalLang)
		fmt.Printf("   %s | %.1f%%\n", notation, confidence)
	}
}

// PrintTopMoves prints the top N predicted moves
func (c *CLI) PrintTopMoves(moves []model.MoveScore, n int) {
	if len(moves) > n {
		moves = moves[:n]
	}

	if !c.quiet {
		fmt.Println("\n📋 Top Predictions:")
		fmt.Println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
	}

	for i, move := range moves {
		c.PrintMoveWithRank(i+1, move)
	}

	if !c.quiet {
		fmt.Println()
	}
}

// PrintStatus prints a status message
func (c *CLI) PrintStatus(message string, level string) {
	if c.quiet && level != "error" {
		return
	}

	var prefix string
	switch level {
	case "info":
		prefix = "ℹ️"
	case "success":
		prefix = "✅"
	case "warning":
		prefix = "⚠️"
	case "error":
		prefix = "❌"
	default:
		prefix = "•"
	}

	fmt.Printf("%s %s\n", prefix, message)
}

// PrintProgress prints a progress indicator
func (c *CLI) PrintProgress(current, total int, message string) {
	if c.quiet {
		return
	}

	percentage := float64(current) / float64(total) * 100
	barLength := 40
	filled := int(float64(barLength) * percentage / 100)

	bar := strings.Repeat("█", filled) + strings.Repeat("░", barLength-filled)
	fmt.Printf("\r[%s] %.1f%% - %s", bar, percentage, message)

	if current >= total {
		fmt.Println()
	}
}

// PrintTrainingStats prints training statistics
func (c *CLI) PrintTrainingStats(epoch, totalEpochs int, loss, accuracy float64, elapsed time.Duration) {
	if c.quiet {
		fmt.Printf("Epoch %d/%d: loss=%.4f, accuracy=%.2f%%, time=%s\n",
			epoch, totalEpochs, loss, accuracy, elapsed)
		return
	}

	fmt.Printf("\n📊 Epoch %d/%d\n", epoch, totalEpochs)
	fmt.Printf("   Loss:     %.4f\n", loss)
	fmt.Printf("   Accuracy: %.2f%%\n", accuracy)
	fmt.Printf("   Time:     %s\n", elapsed)
}

// PrintAnalysisResults prints accuracy analysis results
func (c *CLI) PrintAnalysisResults(totalMoves, correct int, topKAccuracy map[int]float64) {
	accuracy := float64(correct) / float64(totalMoves) * 100

	if c.quiet {
		fmt.Printf("Accuracy: %.2f%% (%d/%d)\n", accuracy, correct, totalMoves)
		for k, acc := range topKAccuracy {
			fmt.Printf("Top-%d: %.2f%%\n", k, acc)
		}
		return
	}

	fmt.Println("\n" + strings.Repeat("━", 70))
	fmt.Println("📊 ANALYSIS RESULTS")
	fmt.Println(strings.Repeat("━", 70))
	fmt.Println()
	fmt.Printf("Total Positions:  %d\n", totalMoves)
	fmt.Printf("Correct Moves:    %d\n", correct)
	fmt.Printf("Top-1 Accuracy:   %.2f%%\n", accuracy)
	fmt.Println()

	if len(topKAccuracy) > 0 {
		fmt.Println("Top-K Accuracy:")
		for _, k := range []int{3, 5, 10} {
			if acc, ok := topKAccuracy[k]; ok {
				fmt.Printf("  Top-%d:  %.2f%%\n", k, acc)
			}
		}
		fmt.Println()
	}

	// Performance rating
	var rating string
	switch {
	case accuracy >= 50:
		rating = "🏆 Excellent - Grandmaster level"
	case accuracy >= 40:
		rating = "🥇 Very Good - Master level"
	case accuracy >= 30:
		rating = "🥈 Good - Expert level"
	case accuracy >= 20:
		rating = "🥉 Fair - Intermediate level"
	default:
		rating = "📚 Learning - Keep training"
	}
	fmt.Printf("Performance: %s\n", rating)
	fmt.Println()
}

// PrintError prints an error message
func (c *CLI) PrintError(err error) {
	fmt.Fprintf(os.Stderr, "❌ Error: %v\n", err)
}

// PrintWarning prints a warning message
func (c *CLI) PrintWarning(message string) {
	if !c.quiet {
		fmt.Printf("⚠️  Warning: %s\n", message)
	}
}

// Helper functions

func (c *CLI) moveToNaturalLanguage(notation string) string {
	if len(notation) < 4 {
		return notation
	}

	from := notation[0:2]
	to := notation[2:4]

	// Try to infer piece type from common patterns
	// This is simplified - a full implementation would need board state
	piece := "piece"

	// Common opening moves
	switch notation {
	case "e2e4":
		return "Move pawn to e4"
	case "e7e5":
		return "Move pawn to e5"
	case "g1f3":
		return "Move knight to f3"
	case "b8c6":
		return "Move knight to c6"
	case "f1c4":
		return "Move bishop to c4"
	case "f8c5":
		return "Move bishop to c5"
	default:
		return fmt.Sprintf("Move %s from %s to %s", piece, from, to)
	}
}

func (c *CLI) getRankEmoji(rank int) string {
	switch rank {
	case 1:
		return "🥇"
	case 2:
		return "🥈"
	case 3:
		return "🥉"
	default:
		return fmt.Sprintf("%d.", rank)
	}
}

// ClearScreen clears the terminal screen
func (c *CLI) ClearScreen() {
	if c.quiet {
		return
	}
	fmt.Print("\033[H\033[2J")
}

// PrintSeparator prints a visual separator
func (c *CLI) PrintSeparator() {
	if !c.quiet {
		fmt.Println(strings.Repeat("━", 70))
	}
}
