package iface

import (
	"fmt"
	"os/exec"
	"runtime"
	"strings"
	"sync"
	"time"

	"go.uber.org/zap"

	"github.com/thyrook/partner/internal/decision"
)

// EnhancedCLI provides rich command-line interface with colors and formatting
type EnhancedCLI struct {
	logger *Logger
	mu     sync.Mutex
	colors bool
}

// NewEnhancedCLI creates a new enhanced CLI
func NewEnhancedCLI(logger *Logger, colors bool) *EnhancedCLI {
	return &EnhancedCLI{
		logger: logger,
		colors: colors,
	}
}

// Color codes for terminal output
const (
	ColorReset  = "\033[0m"
	ColorRed    = "\033[31m"
	ColorGreen  = "\033[32m"
	ColorYellow = "\033[33m"
	ColorBlue   = "\033[34m"
	ColorPurple = "\033[35m"
	ColorCyan   = "\033[36m"
	ColorWhite  = "\033[37m"
	ColorBold   = "\033[1m"
)

// colorize applies color to text if enabled
func (cli *EnhancedCLI) colorize(text, color string) string {
	if !cli.colors {
		return text
	}
	return color + text + ColorReset
}

// PrintBanner displays an enhanced welcome banner
func (cli *EnhancedCLI) PrintBanner() {
	banner := `
╔═══════════════════════════════════════════════════════════╗
║                                                           ║
║     P.A.R.T.N.E.R Chess AI Assistant                    ║
║     Predictive Adaptive Real-Time Neural                 ║
║     Evaluation & Response System                         ║
║                                                           ║
║     Phase 5: Decision Engine & User Interface            ║
║                                                           ║
╚═══════════════════════════════════════════════════════════╝
`
	fmt.Println(cli.colorize(banner, ColorCyan))
}

// PrintDecision displays a complete decision with ranking
func (cli *EnhancedCLI) PrintDecision(dec *decision.Decision) {
	cli.mu.Lock()
	defer cli.mu.Unlock()

	fmt.Println()
	fmt.Println(cli.colorize("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━", ColorCyan))
	fmt.Println(cli.colorize("    🎯 MOVE RECOMMENDATION", ColorBold+ColorWhite))
	fmt.Println(cli.colorize("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━", ColorCyan))
	fmt.Println()

	// Top move
	topMove := dec.TopMove
	moveText := cli.formatMoveNotation(topMove.Move)

	categoryColor := cli.getCategoryColor(topMove.Category)
	fmt.Printf("%s %s\n",
		cli.colorize("Top Move:", ColorBold),
		cli.colorize(moveText, ColorBold+ColorGreen))

	fmt.Printf("%s %.1f%% (%s)\n",
		cli.colorize("Confidence:", ColorWhite),
		topMove.Confidence*100,
		cli.colorize(topMove.Category, categoryColor))

	fmt.Printf("%s %s\n",
		cli.colorize("Explanation:", ColorWhite),
		topMove.Explanation)

	// Alternatives
	if len(dec.Alternatives) > 0 {
		fmt.Println()
		fmt.Println(cli.colorize("Alternative Moves:", ColorBold))

		for i, alt := range dec.Alternatives {
			if i >= 3 {
				break
			}

			altText := cli.formatMoveNotation(alt.Move)
			fmt.Printf("  %d. %s - %.1f%% (%s)\n",
				alt.Rank,
				cli.colorize(altText, ColorYellow),
				alt.Confidence*100,
				alt.Category)
		}
	}

	// Performance metrics
	fmt.Println()
	fmt.Printf("%s %.2f ms | %s %s | %s %d\n",
		cli.colorize("Inference:", ColorBlue),
		dec.InferenceMs,
		cli.colorize("Time:", ColorBlue),
		dec.Timestamp.Format("15:04:05"),
		cli.colorize("Total Options:", ColorBlue),
		dec.TotalMoves)

	fmt.Println()
	fmt.Println(cli.colorize("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━", ColorCyan))
	fmt.Println()

	// Log to file
	cli.logger.Info("Decision displayed",
		zap.String("move", topMove.Move),
		zap.Float64("confidence", topMove.Confidence),
		zap.String("category", topMove.Category),
		zap.Int("alternatives", len(dec.Alternatives)),
	)
}

// formatMoveNotation formats chess notation in readable format
func (cli *EnhancedCLI) formatMoveNotation(move string) string {
	if len(move) != 4 {
		return move
	}

	from := strings.ToUpper(move[0:2])
	to := strings.ToUpper(move[2:4])

	// Convert to "Knight to f3" style format
	return fmt.Sprintf("%s → %s", from, to)
}

// getCategoryColor returns color for move category
func (cli *EnhancedCLI) getCategoryColor(category string) string {
	switch category {
	case "Excellent":
		return ColorGreen
	case "Good":
		return ColorCyan
	case "Fair":
		return ColorYellow
	case "Risky":
		return ColorRed
	default:
		return ColorWhite
	}
}

// PrintStatus displays a status message with timestamp
func (cli *EnhancedCLI) PrintStatus(message string, level string) {
	cli.mu.Lock()
	defer cli.mu.Unlock()

	timestamp := time.Now().Format("15:04:05")

	var icon, color string
	switch level {
	case "info":
		icon = "ℹ"
		color = ColorBlue
	case "success":
		icon = "✓"
		color = ColorGreen
	case "warning":
		icon = "⚠"
		color = ColorYellow
	case "error":
		icon = "✗"
		color = ColorRed
	default:
		icon = "•"
		color = ColorWhite
	}

	fmt.Printf("[%s] %s %s\n",
		cli.colorize(timestamp, ColorBlue),
		cli.colorize(icon, color),
		message)
}

// PrintStatistics displays engine statistics
func (cli *EnhancedCLI) PrintStatistics(stats decision.EngineStats, histStats decision.HistoryStats) {
	cli.mu.Lock()
	defer cli.mu.Unlock()

	fmt.Println()
	fmt.Println(cli.colorize("╔═══════════════════════════════════════════════════════════╗", ColorCyan))
	fmt.Println(cli.colorize("║                  SYSTEM STATISTICS                       ║", ColorCyan))
	fmt.Println(cli.colorize("╚═══════════════════════════════════════════════════════════╝", ColorCyan))
	fmt.Println()

	// Engine stats
	fmt.Println(cli.colorize("  Decision Engine:", ColorBold))
	fmt.Printf("    Total Decisions:     %d\n", stats.TotalDecisions)
	fmt.Printf("    Successful Captures: %d\n", stats.SuccessfulCaptures)
	fmt.Printf("    Failed Captures:     %d\n", stats.FailedCaptures)
	fmt.Printf("    Capture Success:     %.1f%%\n", stats.CaptureSuccessRate)
	fmt.Printf("    Avg Inference:       %.2f ms\n", stats.AvgInferenceMs)
	fmt.Printf("    Total Inference:     %v\n", stats.TotalInferenceTime.Round(time.Millisecond))

	// History stats
	if histStats.TotalDecisions > 0 {
		fmt.Println()
		fmt.Println(cli.colorize("  Decision History:", ColorBold))
		fmt.Printf("    Total Recorded:      %d\n", histStats.TotalDecisions)
		fmt.Printf("    Avg Confidence:      %.1f%%\n", histStats.AvgConfidence*100)
		fmt.Printf("    Avg Inference:       %.2f ms\n", histStats.AvgInferenceMs)

		if len(histStats.CategoryCounts) > 0 {
			fmt.Println()
			fmt.Println(cli.colorize("  Decision Categories:", ColorBold))
			for category, count := range histStats.CategoryCounts {
				percentage := float64(count) / float64(histStats.TotalDecisions) * 100
				color := cli.getCategoryColor(category)
				fmt.Printf("    %s: %d (%.1f%%)\n",
					cli.colorize(category, color),
					count,
					percentage)
			}
		}
	}

	fmt.Println()
	fmt.Println(cli.colorize("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━", ColorCyan))
	fmt.Println()
}

// PrintError displays an error with context
func (cli *EnhancedCLI) PrintError(err error, context string) {
	cli.mu.Lock()
	defer cli.mu.Unlock()

	fmt.Printf("%s %s: %v\n",
		cli.colorize("✗", ColorRed),
		cli.colorize(context, ColorRed),
		err)

	cli.logger.Error(context, zap.Error(err))
}

// PrintProgress displays a progress bar
func (cli *EnhancedCLI) PrintProgress(current, total int, label string) {
	cli.mu.Lock()
	defer cli.mu.Unlock()

	percent := float64(current) / float64(total) * 100
	width := 40
	filled := int(percent / 100 * float64(width))

	bar := "["
	for i := 0; i < width; i++ {
		if i < filled {
			bar += "█"
		} else {
			bar += "░"
		}
	}
	bar += "]"

	fmt.Printf("\r%s %s %.1f%% (%d/%d)",
		label,
		cli.colorize(bar, ColorGreen),
		percent,
		current,
		total)

	if current >= total {
		fmt.Println()
	}
}

// EnhancedVoice provides text-to-speech with multiple engine support
type EnhancedVoice struct {
	enabled  bool
	engine   string // "espeak", "say", "none"
	logger   *zap.Logger
	mu       sync.Mutex
	queue    []string
	speaking bool
}

// NewEnhancedVoice creates a new enhanced voice system
func NewEnhancedVoice(enabled bool, logger *zap.Logger) *EnhancedVoice {
	voice := &EnhancedVoice{
		enabled: enabled,
		logger:  logger,
		queue:   make([]string, 0),
	}

	// Detect available TTS engine
	voice.engine = voice.detectTTSEngine()

	if enabled && voice.engine == "none" {
		logger.Warn("TTS requested but no engine found",
			zap.String("os", runtime.GOOS))
	}

	return voice
}

// detectTTSEngine detects available TTS engine
func (ev *EnhancedVoice) detectTTSEngine() string {
	switch runtime.GOOS {
	case "darwin":
		// macOS - check for 'say' command
		if _, err := exec.LookPath("say"); err == nil {
			return "say"
		}
	case "linux":
		// Linux - check for espeak
		if _, err := exec.LookPath("espeak"); err == nil {
			return "espeak"
		}
		// Try espeak-ng
		if _, err := exec.LookPath("espeak-ng"); err == nil {
			return "espeak-ng"
		}
	}
	return "none"
}

// Speak announces a message using TTS
func (ev *EnhancedVoice) Speak(message string) error {
	if !ev.enabled || ev.engine == "none" {
		return nil
	}

	ev.mu.Lock()
	defer ev.mu.Unlock()

	// Add to queue
	ev.queue = append(ev.queue, message)

	// Process queue if not speaking
	if !ev.speaking {
		go ev.processQueue()
	}

	return nil
}

// processQueue processes the speech queue
func (ev *EnhancedVoice) processQueue() {
	ev.mu.Lock()
	ev.speaking = true
	ev.mu.Unlock()

	defer func() {
		ev.mu.Lock()
		ev.speaking = false
		ev.mu.Unlock()
	}()

	for {
		ev.mu.Lock()
		if len(ev.queue) == 0 {
			ev.mu.Unlock()
			return
		}

		message := ev.queue[0]
		ev.queue = ev.queue[1:]
		ev.mu.Unlock()

		if err := ev.speakNow(message); err != nil {
			ev.logger.Error("TTS failed",
				zap.Error(err),
				zap.String("engine", ev.engine))
		}
	}
}

// speakNow executes TTS immediately
func (ev *EnhancedVoice) speakNow(message string) error {
	var cmd *exec.Cmd

	switch ev.engine {
	case "say":
		// macOS
		cmd = exec.Command("say", message)
	case "espeak":
		// Linux espeak
		cmd = exec.Command("espeak", message)
	case "espeak-ng":
		// Linux espeak-ng
		cmd = exec.Command("espeak-ng", message)
	default:
		return fmt.Errorf("no TTS engine available")
	}

	ev.logger.Debug("TTS speaking",
		zap.String("engine", ev.engine),
		zap.String("message", message))

	return cmd.Run()
}

// AnnounceDecision announces a decision via TTS
func (ev *EnhancedVoice) AnnounceDecision(dec *decision.Decision) error {
	if !ev.enabled {
		return nil
	}

	move := ev.formatMoveForSpeech(dec.TopMove.Move)
	confidence := int(dec.TopMove.Confidence * 100)
	category := dec.TopMove.Category

	var message string
	if confidence >= 80 {
		message = fmt.Sprintf("Strong move: %s. Confidence %d percent.", move, confidence)
	} else if confidence >= 60 {
		message = fmt.Sprintf("Recommended move: %s. Confidence %d percent.", move, confidence)
	} else {
		message = fmt.Sprintf("Consider move: %s. %s confidence, %d percent.",
			move, category, confidence)
	}

	return ev.Speak(message)
}

// formatMoveForSpeech converts move to speakable format
func (ev *EnhancedVoice) formatMoveForSpeech(move string) string {
	if len(move) != 4 {
		return move
	}

	// Convert "e2e4" to "e2 to e4"
	return fmt.Sprintf("%c %c to %c %c",
		move[0], move[1], move[2], move[3])
}

// AnnounceStat istics announces key statistics
func (ev *EnhancedVoice) AnnounceStatistics(stats decision.EngineStats) error {
	if !ev.enabled {
		return nil
	}

	message := fmt.Sprintf("System statistics. Total decisions: %d. "+
		"Capture success rate: %.0f percent. "+
		"Average inference time: %.0f milliseconds.",
		stats.TotalDecisions,
		stats.CaptureSuccessRate,
		stats.AvgInferenceMs)

	return ev.Speak(message)
}

// Stop stops any ongoing speech
func (ev *EnhancedVoice) Stop() {
	ev.mu.Lock()
	defer ev.mu.Unlock()

	ev.queue = make([]string, 0)

	// Platform-specific stop commands
	switch ev.engine {
	case "say":
		exec.Command("killall", "say").Run()
	case "espeak", "espeak-ng":
		exec.Command("killall", ev.engine).Run()
	}
}

// IsEnabled returns whether voice is enabled
func (ev *EnhancedVoice) IsEnabled() bool {
	return ev.enabled && ev.engine != "none"
}

// GetEngine returns the TTS engine in use
func (ev *EnhancedVoice) GetEngine() string {
	return ev.engine
}
