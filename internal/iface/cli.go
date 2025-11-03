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
ANALYZE MODE
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

	fmt.Printf("\nEpoch %d/%d\n", epoch, totalEpochs)
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
	fmt.Println("ANALYSIS RESULTS")
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
	fmt.Fprintf(os.Stderr, "Error: %v\n", err)
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
	fmt.Print("\033[2J\033[H") // ANSI escape code to clear screen
}

// ===== ENHANCED FORMATTING HELPERS =====

// Color codes for terminal output
const (
	ColorReset   = "\033[0m"
	ColorRed     = "\033[31m"
	ColorGreen   = "\033[32m"
	ColorYellow  = "\033[33m"
	ColorBlue    = "\033[34m"
	ColorMagenta = "\033[35m"
	ColorCyan    = "\033[36m"
	ColorWhite   = "\033[37m"
	ColorBold    = "\033[1m"
	ColorDim     = "\033[2m"
)

// Colorize applies color to text if terminal supports it
func (c *CLI) Colorize(text string, color string) string {
	if c.quiet || os.Getenv("NO_COLOR") != "" {
		return text
	}
	return color + text + ColorReset
}

// PrintSuccess prints a success message in green
func (c *CLI) PrintSuccess(message string) {
	if !c.quiet {
		fmt.Println(c.Colorize("✓ "+message, ColorGreen))
	}
}

// PrintInfo prints an info message in blue
func (c *CLI) PrintInfo(message string) {
	if !c.quiet {
		fmt.Println(c.Colorize("ℹ "+message, ColorBlue))
	}
}

// PrintProgressBar displays a progress bar
func (c *CLI) PrintProgressBar(current, total int, label string) {
	if c.quiet || total == 0 {
		return
	}

	width := 40
	percentage := float64(current) / float64(total)
	filled := int(percentage * float64(width))

	bar := "["
	for i := 0; i < width; i++ {
		if i < filled {
			bar += "█"
		} else {
			bar += "░"
		}
	}
	bar += "]"

	fmt.Printf("\r%s %s %d/%d (%.1f%%) ", label, bar, current, total, percentage*100)
	if current == total {
		fmt.Println() // New line when complete
	}
}

// PrintTable prints data in a formatted table
func (c *CLI) PrintTable(headers []string, rows [][]string) {
	if c.quiet {
		return
	}

	// Calculate column widths
	colWidths := make([]int, len(headers))
	for i, h := range headers {
		colWidths[i] = len(h)
	}
	for _, row := range rows {
		for i, cell := range row {
			if i < len(colWidths) && len(cell) > colWidths[i] {
				colWidths[i] = len(cell)
			}
		}
	}

	// Print header
	fmt.Println()
	for i, h := range headers {
		fmt.Printf("%-*s  ", colWidths[i], h)
	}
	fmt.Println()

	// Print separator
	for _, w := range colWidths {
		fmt.Print(strings.Repeat("─", w+2))
	}
	fmt.Println()

	// Print rows
	for _, row := range rows {
		for i, cell := range row {
			if i < len(colWidths) {
				fmt.Printf("%-*s  ", colWidths[i], cell)
			}
		}
		fmt.Println()
	}
	fmt.Println()
}

// PrintBox prints text in a box
func (c *CLI) PrintBox(title string, lines []string) {
	if c.quiet {
		return
	}

	// Find max width
	maxWidth := len(title)
	for _, line := range lines {
		if len(line) > maxWidth {
			maxWidth = len(line)
		}
	}

	width := maxWidth + 4 // Padding

	// Top border
	fmt.Println("┌" + strings.Repeat("─", width) + "┐")

	// Title
	padding := (width - len(title)) / 2
	fmt.Printf("│%s%s%s│\n",
		strings.Repeat(" ", padding),
		c.Colorize(title, ColorBold),
		strings.Repeat(" ", width-padding-len(title)))

	// Separator
	fmt.Println("├" + strings.Repeat("─", width) + "┤")

	// Content lines
	for _, line := range lines {
		fmt.Printf("│ %-*s │\n", width-2, line)
	}

	// Bottom border
	fmt.Println("└" + strings.Repeat("─", width) + "┘")
}

// PrintSpinner displays an animated spinner (for long operations)
func (c *CLI) PrintSpinner(message string, done chan bool) {
	if c.quiet {
		return
	}

	spinChars := []string{"⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"}
	i := 0

	for {
		select {
		case <-done:
			fmt.Printf("\r%s\n", strings.Repeat(" ", len(message)+4))
			return
		default:
			fmt.Printf("\r%s %s ", spinChars[i%len(spinChars)], message)
			time.Sleep(80 * time.Millisecond)
			i++
		}
	}
}

// PrintStats prints statistics in a formatted grid
func (c *CLI) PrintStats(stats map[string]interface{}) {
	if c.quiet {
		return
	}

	fmt.Println("\n" + strings.Repeat("═", 70))
	fmt.Println(c.Colorize(" STATISTICS", ColorBold+ColorCyan))
	fmt.Println(strings.Repeat("═", 70))

	for key, value := range stats {
		fmt.Printf("  %-30s: ", key)

		// Format value based on type
		switch v := value.(type) {
		case float64:
			if v >= 1000 {
				fmt.Printf(c.Colorize("%.2f", ColorGreen), v)
			} else if v >= 100 {
				fmt.Printf(c.Colorize("%.2f", ColorYellow), v)
			} else {
				fmt.Printf("%.2f", v)
			}
		case int:
			if v >= 1000 {
				fmt.Printf(c.Colorize("%d", ColorGreen), v)
			} else {
				fmt.Printf("%d", v)
			}
		case bool:
			if v {
				fmt.Print(c.Colorize("✓ Yes", ColorGreen))
			} else {
				fmt.Print(c.Colorize("✗ No", ColorRed))
			}
		case string:
			fmt.Print(v)
		default:
			fmt.Printf("%v", v)
		}
		fmt.Println()
	}

	fmt.Println(strings.Repeat("═", 70) + "\n")
}

// PrintHelp prints comprehensive help information
func (c *CLI) PrintHelp(commandName string) {
	switch commandName {
	case "train":
		c.PrintBox("TRAINING HELP", []string{
			"Usage: partner train [options]",
			"",
			"Options:",
			"  --epochs N      Number of training epochs (default: 50)",
			"  --batch-size N  Batch size (default: 64)",
			"  --lr FLOAT      Learning rate (default: 0.001)",
			"  --data PATH     Path to dataset (default: data/positions.db)",
			"  --model PATH    Path to save model (default: data/models/chess_cnn.gob)",
			"",
			"Examples:",
			"  partner train --epochs 100 --batch-size 128",
			"  partner train --data custom.db --lr 0.0001",
		})

	case "infer":
		c.PrintBox("INFERENCE HELP", []string{
			"Usage: partner infer [options]",
			"",
			"Options:",
			"  --model PATH    Path to trained model (required)",
			"  --fen FEN       FEN string to analyze",
			"  --batch N       Run batch inference on N random positions",
			"  --top-k N       Show top N moves (default: 5)",
			"",
			"Examples:",
			"  partner infer --model chess.gob --fen \"rnbqkbnr/pppppppp/...\"",
			"  partner infer --batch 100 --top-k 3",
		})

	case "dataset":
		c.PrintBox("DATASET HELP", []string{
			"Usage: partner dataset [command]",
			"",
			"Commands:",
			"  stats           Show dataset statistics",
			"  ingest FILE     Ingest PGN file into dataset",
			"  export FILE     Export samples to file",
			"  validate        Validate dataset integrity",
			"  compact         Compact database",
			"",
			"Examples:",
			"  partner dataset stats",
			"  partner dataset ingest games.pgn",
			"  partner dataset export samples.txt --count 1000",
		})

	default:
		c.PrintBox("P.A.R.T.N.E.R HELP", []string{
			"Pattern Analysis & Real-Time Neural Enhancement for Reinforcement",
			"",
			"Available Commands:",
			"  train      Train the neural network on chess positions",
			"  infer      Run inference on chess positions",
			"  dataset    Manage training dataset",
			"  status     Show system status",
			"  help       Show help for specific command",
			"",
			"Usage:",
			"  partner <command> [options]",
			"",
			"Get help for specific command:",
			"  partner help <command>",
			"",
			"Examples:",
			"  partner train --epochs 100",
			"  partner infer --model chess.gob --batch 50",
			"  partner dataset stats",
		})
	}
}

// PrintSeparator prints a visual separator
func (c *CLI) PrintSeparator() {
	if !c.quiet {
		fmt.Println(strings.Repeat("━", 70))
	}
}
