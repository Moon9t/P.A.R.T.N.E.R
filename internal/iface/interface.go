package iface

import (
	"fmt"
	"io"
	"os"
	"strings"
	"sync"
	"time"

	"go.uber.org/zap"
	"go.uber.org/zap/zapcore"

	"github.com/thyrook/partner/internal/decision"
)

// Logger wraps zap logger for structured logging
type Logger struct {
	zap *zap.Logger
	mu  sync.Mutex
}

// NewLogger creates a new logger instance
func NewLogger(logPath string, level string) (*Logger, error) {
	// Create log directory if it doesn't exist
	if err := os.MkdirAll("logs", 0755); err != nil {
		return nil, fmt.Errorf("failed to create log directory: %w", err)
	}

	// Parse log level
	var zapLevel zapcore.Level
	switch level {
	case "debug":
		zapLevel = zapcore.DebugLevel
	case "info":
		zapLevel = zapcore.InfoLevel
	case "warn":
		zapLevel = zapcore.WarnLevel
	case "error":
		zapLevel = zapcore.ErrorLevel
	default:
		zapLevel = zapcore.InfoLevel
	}

	// Configure encoder
	encoderConfig := zapcore.EncoderConfig{
		TimeKey:        "time",
		LevelKey:       "level",
		NameKey:        "logger",
		CallerKey:      "caller",
		MessageKey:     "msg",
		StacktraceKey:  "stacktrace",
		LineEnding:     zapcore.DefaultLineEnding,
		EncodeLevel:    zapcore.CapitalLevelEncoder,
		EncodeTime:     zapcore.ISO8601TimeEncoder,
		EncodeDuration: zapcore.StringDurationEncoder,
		EncodeCaller:   zapcore.ShortCallerEncoder,
	}

	// Create file writer
	logFile, err := os.OpenFile(logPath, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		return nil, fmt.Errorf("failed to open log file: %w", err)
	}

	// Create multi-writer (console + file)
	multiWriter := io.MultiWriter(os.Stdout, logFile)

	core := zapcore.NewCore(
		zapcore.NewConsoleEncoder(encoderConfig),
		zapcore.AddSync(multiWriter),
		zapLevel,
	)

	logger := zap.New(core, zap.AddCaller())

	return &Logger{
		zap: logger,
	}, nil
}

// Info logs an info message
func (l *Logger) Info(msg string, fields ...zap.Field) {
	l.mu.Lock()
	defer l.mu.Unlock()
	l.zap.Info(msg, fields...)
}

// Error logs an error message
func (l *Logger) Error(msg string, fields ...zap.Field) {
	l.mu.Lock()
	defer l.mu.Unlock()
	l.zap.Error(msg, fields...)
}

// Warn logs a warning message
func (l *Logger) Warn(msg string, fields ...zap.Field) {
	l.mu.Lock()
	defer l.mu.Unlock()
	l.zap.Warn(msg, fields...)
}

// Debug logs a debug message
func (l *Logger) Debug(msg string, fields ...zap.Field) {
	l.mu.Lock()
	defer l.mu.Unlock()
	l.zap.Debug(msg, fields...)
}

// Sync flushes buffered logs
func (l *Logger) Sync() error {
	return l.zap.Sync()
}

// GetZapLogger returns the underlying zap logger
func (l *Logger) GetZapLogger() *zap.Logger {
	return l.zap
}

// CLI handles command-line interface interactions
type CLI struct {
	logger *Logger
	mu     sync.Mutex
}

// NewCLI creates a new CLI instance
func NewCLI(logger *Logger) *CLI {
	return &CLI{
		logger: logger,
	}
}

// PrintWelcome displays the welcome message
func (c *CLI) PrintWelcome() {
	fmt.Println(strings.Repeat("=", 60))
	fmt.Println("  P.A.R.T.N.E.R - Predictive Adaptive Real-Time Neural")
	fmt.Println("           Evaluation & Response System")
	fmt.Println(strings.Repeat("=", 60))
	fmt.Println()
}

// PrintStatus displays current system status
func (c *CLI) PrintStatus(status string) {
	c.mu.Lock()
	defer c.mu.Unlock()

	timestamp := time.Now().Format("15:04:05")
	fmt.Printf("[%s] %s\n", timestamp, status)
}

// PrintAdvice displays move advice
func (c *CLI) PrintAdvice(advice *decision.Advice) {
	c.mu.Lock()
	defer c.mu.Unlock()

	fmt.Println("\n" + strings.Repeat("─", 50))
	fmt.Printf("🎯 MOVE SUGGESTION\n")
	fmt.Println(strings.Repeat("─", 50))

	formatted := decision.FormatAdvice(advice)
	fmt.Println(formatted)

	fmt.Println(strings.Repeat("─", 50) + "\n")

	// Log to file
	c.logger.Info("Move suggested",
		zap.String("move", advice.PrimaryMove),
		zap.Float64("confidence", advice.Confidence),
	)
}

// PrintPrediction displays a prediction result
func (c *CLI) PrintPrediction(pred *decision.Prediction) {
	c.mu.Lock()
	defer c.mu.Unlock()

	fmt.Printf("\n➤ Predicted Move: %s (Confidence: %.1f%%)\n",
		pred.Move, pred.Confidence*100)
}

// PrintError displays an error message
func (c *CLI) PrintError(err error) {
	c.mu.Lock()
	defer c.mu.Unlock()

	fmt.Printf("❌ Error: %v\n", err)
	c.logger.Error("Error occurred", zap.Error(err))
}

// PrintTrainingProgress displays training progress
func (c *CLI) PrintTrainingProgress(epoch int, loss float64, bufferSize int) {
	c.mu.Lock()
	defer c.mu.Unlock()

	fmt.Printf("Training Epoch %d: Loss=%.4f, Buffer=%d samples\n", epoch, loss, bufferSize)
}

// PromptUser prompts for user input
func (c *CLI) PromptUser(prompt string) string {
	c.mu.Lock()
	defer c.mu.Unlock()

	fmt.Print(prompt)
	var input string
	fmt.Scanln(&input)
	return input
}

// PrintMenu displays a menu
func (c *CLI) PrintMenu(options []string) {
	c.mu.Lock()
	defer c.mu.Unlock()

	fmt.Println("\nSelect an option:")
	for i, option := range options {
		fmt.Printf("  %d. %s\n", i+1, option)
	}
	fmt.Print("\nChoice: ")
}

// VoiceFeedback handles text-to-speech output (optional)
type VoiceFeedback struct {
	enabled bool
	logger  *Logger
}

// NewVoiceFeedback creates a new voice feedback handler
func NewVoiceFeedback(enabled bool, logger *Logger) *VoiceFeedback {
	return &VoiceFeedback{
		enabled: enabled,
		logger:  logger,
	}
}

// Speak announces a message (placeholder for TTS integration)
func (vf *VoiceFeedback) Speak(message string) {
	if !vf.enabled {
		return
	}

	// Placeholder: In a real implementation, this would call a TTS API
	// For now, we just log it
	vf.logger.Info("Voice announcement", zap.String("message", message))

	// Example integration point for TTS libraries:
	// - espeak (Linux): exec.Command("espeak", message)
	// - say (macOS): exec.Command("say", message)
	// - Cloud TTS API: Google Cloud TTS, Amazon Polly, etc.
}

// AnnounceMove announces a suggested move
func (vf *VoiceFeedback) AnnounceMove(move string, confidence float64) {
	message := fmt.Sprintf("Consider moving %s with %.0f percent confidence",
		vf.formatMoveForSpeech(move), confidence*100)
	vf.Speak(message)
}

// formatMoveForSpeech converts chess notation to speakable format
func (vf *VoiceFeedback) formatMoveForSpeech(move string) string {
	if len(move) != 4 {
		return move
	}

	// Convert "e2e4" to "e2 to e4"
	return fmt.Sprintf("%c%c to %c%c", move[0], move[1], move[2], move[3])
}

// ProgressBar displays a progress bar
type ProgressBar struct {
	total   int
	current int
	width   int
	mu      sync.Mutex
}

// NewProgressBar creates a new progress bar
func NewProgressBar(total, width int) *ProgressBar {
	return &ProgressBar{
		total: total,
		width: width,
	}
}

// Update updates the progress bar
func (pb *ProgressBar) Update(current int) {
	pb.mu.Lock()
	defer pb.mu.Unlock()

	pb.current = current
	pb.render()
}

// render displays the progress bar
func (pb *ProgressBar) render() {
	percent := float64(pb.current) / float64(pb.total)
	filled := int(percent * float64(pb.width))

	bar := "["
	for i := 0; i < pb.width; i++ {
		if i < filled {
			bar += "="
		} else if i == filled {
			bar += ">"
		} else {
			bar += " "
		}
	}
	bar += fmt.Sprintf("] %.1f%%", percent*100)

	fmt.Printf("\r%s", bar)

	if pb.current >= pb.total {
		fmt.Println()
	}
}

// Statistics displays system statistics
type Statistics struct {
	cli *CLI
}

// NewStatistics creates a new statistics display
func NewStatistics(cli *CLI) *Statistics {
	return &Statistics{cli: cli}
}

// DisplayStats shows system statistics
func (s *Statistics) DisplayStats(stats decision.Statistics) {
	fmt.Println("\n" + strings.Repeat("═", 50))
	fmt.Println("  SYSTEM STATISTICS")
	fmt.Println(strings.Repeat("═", 50))
	fmt.Printf("Total Predictions: %d\n", stats.TotalPredictions)
	fmt.Printf("Average Confidence: %.1f%%\n", stats.AverageConfidence*100)
	fmt.Println(strings.Repeat("═", 50) + "\n")
}

// SessionInfo displays session information
type SessionInfo struct {
	StartTime       time.Time
	Predictions     int
	TrainingSamples int
}

// NewSessionInfo creates a new session info tracker
func NewSessionInfo() *SessionInfo {
	return &SessionInfo{
		StartTime: time.Now(),
	}
}

// Display shows session information
func (si *SessionInfo) Display() {
	duration := time.Since(si.StartTime)

	fmt.Println("\n" + strings.Repeat("─", 50))
	fmt.Println("SESSION INFO")
	fmt.Println(strings.Repeat("─", 50))
	fmt.Printf("Session Duration: %v\n", duration.Round(time.Second))
	fmt.Printf("Predictions Made: %d\n", si.Predictions)
	fmt.Printf("Training Samples: %d\n", si.TrainingSamples)
	fmt.Println(strings.Repeat("─", 50) + "\n")
}

// UpdatePredictions increments prediction counter
func (si *SessionInfo) UpdatePredictions() {
	si.Predictions++
}

// UpdateTrainingSamples updates training sample counter
func (si *SessionInfo) UpdateTrainingSamples(count int) {
	si.TrainingSamples = count
}
