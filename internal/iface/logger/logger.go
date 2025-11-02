package logger

import (
	"fmt"
	"io"
	"log/slog"
	"os"
	"path/filepath"
)

var (
	// Default logger instance
	defaultLogger *slog.Logger
)

// Level represents log level
type Level string

const (
	LevelDebug Level = "debug"
	LevelInfo  Level = "info"
	LevelWarn  Level = "warn"
	LevelError Level = "error"
)

// Setup initializes the logger with the specified configuration
func Setup(level Level, logPath string) error {
	// Parse log level
	var slogLevel slog.Level
	switch level {
	case LevelDebug:
		slogLevel = slog.LevelDebug
	case LevelInfo:
		slogLevel = slog.LevelInfo
	case LevelWarn:
		slogLevel = slog.LevelWarn
	case LevelError:
		slogLevel = slog.LevelError
	default:
		slogLevel = slog.LevelInfo
	}

	// Create log directory if it doesn't exist
	if logPath != "" {
		dir := filepath.Dir(logPath)
		if err := os.MkdirAll(dir, 0755); err != nil {
			return fmt.Errorf("failed to create log directory: %w", err)
		}
	}

	var writers []io.Writer
	writers = append(writers, os.Stdout)

	// Add file writer if log path is specified
	if logPath != "" {
		file, err := os.OpenFile(logPath, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0644)
		if err != nil {
			return fmt.Errorf("failed to open log file: %w", err)
		}
		writers = append(writers, file)
	}

	// Create multi-writer
	multiWriter := io.MultiWriter(writers...)

	// Create handler with options
	opts := &slog.HandlerOptions{
		Level:     slogLevel,
		AddSource: false,
	}

	handler := slog.NewTextHandler(multiWriter, opts)
	defaultLogger = slog.New(handler)

	return nil
}

// Get returns the default logger instance
func Get() *slog.Logger {
	if defaultLogger == nil {
		// Initialize with default settings if not set up
		Setup(LevelInfo, "")
	}
	return defaultLogger
}

// Debug logs a debug message
func Debug(msg string, args ...any) {
	Get().Debug(msg, args...)
}

// Info logs an info message
func Info(msg string, args ...any) {
	Get().Info(msg, args...)
}

// Warn logs a warning message
func Warn(msg string, args ...any) {
	Get().Warn(msg, args...)
}

// Error logs an error message
func Error(msg string, args ...any) {
	Get().Error(msg, args...)
}

// With returns a logger with additional attributes
func With(args ...any) *slog.Logger {
	return Get().With(args...)
}

// WithGroup returns a logger with a group name
func WithGroup(name string) *slog.Logger {
	return Get().WithGroup(name)
}

// LogEvent logs a structured event
func LogEvent(event string, attrs map[string]any) {
	args := make([]any, 0, len(attrs)*2)
	for k, v := range attrs {
		args = append(args, k, v)
	}
	Get().Info(event, args...)
}

// LogPerformance logs performance metrics
func LogPerformance(operation string, duration float64, success bool) {
	Get().Info("performance",
		"operation", operation,
		"duration_ms", duration,
		"success", success,
	)
}

// LogError logs an error with context
func LogError(err error, context map[string]any) {
	args := make([]any, 0, len(context)*2+2)
	args = append(args, "error", err.Error())
	for k, v := range context {
		args = append(args, k, v)
	}
	Get().Error("error occurred", args...)
}
