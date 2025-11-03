package logger

import (
	"bufio"
	"fmt"
	"io"
	"log/slog"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"
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

// ===== ENHANCED LOGGING HELPERS =====

// TrainingMetrics holds training-specific metrics
type TrainingMetrics struct {
	Epoch          int
	Loss           float64
	Accuracy       float64
	ValidationLoss float64
	ValAccuracy    float64
	LearningRate   float64
	BatchSize      int
	ThroughputSPS  float64 // Samples per second
	TimeElapsed    float64 // Seconds
}

// LogTraining logs training progress with rich metrics
func LogTraining(metrics TrainingMetrics) {
	Get().Info("training_epoch",
		"epoch", metrics.Epoch,
		"loss", fmt.Sprintf("%.4f", metrics.Loss),
		"accuracy", fmt.Sprintf("%.2f%%", metrics.Accuracy),
		"val_loss", fmt.Sprintf("%.4f", metrics.ValidationLoss),
		"val_acc", fmt.Sprintf("%.2f%%", metrics.ValAccuracy),
		"lr", fmt.Sprintf("%.6f", metrics.LearningRate),
		"batch_size", metrics.BatchSize,
		"throughput", fmt.Sprintf("%.1f samples/s", metrics.ThroughputSPS),
		"time", fmt.Sprintf("%.2fs", metrics.TimeElapsed),
	)
}

// InferenceMetrics holds inference-specific metrics
type InferenceMetrics struct {
	Move         string
	Confidence   float64
	Rank         int
	LatencyMs    float64
	TopKAccuracy float64
	FromCache    bool
}

// LogInference logs model inference with detailed metrics
func LogInference(metrics InferenceMetrics) {
	Get().Info("model_inference",
		"move", metrics.Move,
		"confidence", fmt.Sprintf("%.2f%%", metrics.Confidence*100),
		"rank", metrics.Rank,
		"latency_ms", fmt.Sprintf("%.2f", metrics.LatencyMs),
		"top_k_acc", fmt.Sprintf("%.2f%%", metrics.TopKAccuracy*100),
		"cached", metrics.FromCache,
	)
}

// DatasetMetrics holds dataset operation metrics
type DatasetMetrics struct {
	Operation      string // "ingest", "load", "sample"
	TotalPositions int64
	ProcessedCount int64
	SuccessRate    float64
	TimeElapsed    float64
	ThroughputPPS  float64 // Positions per second
	ErrorCount     int
}

// LogDataset logs dataset operations with statistics
func LogDataset(metrics DatasetMetrics) {
	Get().Info("dataset_operation",
		"operation", metrics.Operation,
		"total", metrics.TotalPositions,
		"processed", metrics.ProcessedCount,
		"success_rate", fmt.Sprintf("%.2f%%", metrics.SuccessRate),
		"time", fmt.Sprintf("%.2fs", metrics.TimeElapsed),
		"throughput", fmt.Sprintf("%.1f pos/s", metrics.ThroughputPPS),
		"errors", metrics.ErrorCount,
	)
}

// SystemMetrics holds system resource metrics
type SystemMetrics struct {
	CPUPercent     float64
	MemoryUsedMB   float64
	MemoryTotalMB  float64
	GPUUtilization float64 // If available
	DiskUsedGB     float64
	ActiveThreads  int
}

// LogSystemMetrics logs system resource usage
func LogSystemMetrics(metrics SystemMetrics) {
	Get().Info("system_metrics",
		"cpu_percent", fmt.Sprintf("%.1f%%", metrics.CPUPercent),
		"memory_used_mb", fmt.Sprintf("%.1f", metrics.MemoryUsedMB),
		"memory_total_mb", fmt.Sprintf("%.1f", metrics.MemoryTotalMB),
		"memory_percent", fmt.Sprintf("%.1f%%", (metrics.MemoryUsedMB/metrics.MemoryTotalMB)*100),
		"gpu_util", fmt.Sprintf("%.1f%%", metrics.GPUUtilization),
		"disk_used_gb", fmt.Sprintf("%.1f", metrics.DiskUsedGB),
		"threads", metrics.ActiveThreads,
	)
}

// GameMetrics holds game-specific metrics
type GameMetrics struct {
	GameType      string // "chess", "racing", etc.
	MovesPlayed   int
	AvgConfidence float64
	CorrectMoves  int
	TotalMoves    int
	GameDuration  float64 // Seconds
	Result        string  // "win", "loss", "draw"
}

// LogGameSession logs game session statistics
func LogGameSession(metrics GameMetrics) {
	accuracy := 0.0
	if metrics.TotalMoves > 0 {
		accuracy = (float64(metrics.CorrectMoves) / float64(metrics.TotalMoves)) * 100
	}

	Get().Info("game_session",
		"game_type", metrics.GameType,
		"moves", metrics.MovesPlayed,
		"avg_confidence", fmt.Sprintf("%.2f%%", metrics.AvgConfidence*100),
		"accuracy", fmt.Sprintf("%.2f%%", accuracy),
		"duration", fmt.Sprintf("%.1fs", metrics.GameDuration),
		"result", metrics.Result,
	)
}

// ProfilerMetrics tracks code performance profiling
type ProfilerMetrics struct {
	FunctionName string
	CallCount    int
	TotalTimeMs  float64
	AvgTimeMs    float64
	MinTimeMs    float64
	MaxTimeMs    float64
}

// LogProfiler logs performance profiling data
func LogProfiler(metrics ProfilerMetrics) {
	Get().Debug("profiler",
		"function", metrics.FunctionName,
		"calls", metrics.CallCount,
		"total_ms", fmt.Sprintf("%.2f", metrics.TotalTimeMs),
		"avg_ms", fmt.Sprintf("%.3f", metrics.AvgTimeMs),
		"min_ms", fmt.Sprintf("%.3f", metrics.MinTimeMs),
		"max_ms", fmt.Sprintf("%.3f", metrics.MaxTimeMs),
	)
}

// StartOperation logs the start of an operation (returns cleanup function)
func StartOperation(operation string, attrs map[string]any) func(error) {
	startTime := time.Now()

	logArgs := make([]any, 0, len(attrs)*2+2)
	logArgs = append(logArgs, "operation", operation)
	for k, v := range attrs {
		logArgs = append(logArgs, k, v)
	}
	Get().Info("operation_start", logArgs...)

	// Return cleanup function
	return func(err error) {
		duration := time.Since(startTime)
		success := err == nil

		endArgs := make([]any, 0, len(attrs)*2+6)
		endArgs = append(endArgs, "operation", operation)
		endArgs = append(endArgs, "duration_ms", duration.Milliseconds())
		endArgs = append(endArgs, "success", success)
		if err != nil {
			endArgs = append(endArgs, "error", err.Error())
		}
		for k, v := range attrs {
			endArgs = append(endArgs, k, v)
		}

		if success {
			Get().Info("operation_complete", endArgs...)
		} else {
			Get().Error("operation_failed", endArgs...)
		}
	}
}

// LogSummary represents aggregated log statistics
type LogSummary struct {
	TotalEntries int
	ErrorCount   int
	WarningCount int
	InfoCount    int
	DebugCount   int
	AvgLatencyMs float64
	ErrorRate    float64
	TopErrors    map[string]int
}

// AnalyzeLogs provides summary statistics by parsing log files
func AnalyzeLogs(logFilePath string) (*LogSummary, error) {
	file, err := os.Open(logFilePath)
	if err != nil {
		return nil, fmt.Errorf("failed to open log file: %w", err)
	}
	defer file.Close()

	summary := &LogSummary{
		TopErrors: make(map[string]int),
	}

	var totalLatency float64
	var latencyCount int

	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		line := scanner.Text()
		summary.TotalEntries++

		// Count log levels
		if strings.Contains(line, "level=ERROR") {
			summary.ErrorCount++
			// Extract error messages
			if idx := strings.Index(line, "error="); idx != -1 {
				errorMsg := extractQuotedValue(line[idx:])
				if errorMsg != "" {
					summary.TopErrors[errorMsg]++
				}
			}
		} else if strings.Contains(line, "level=WARN") {
			summary.WarningCount++
		} else if strings.Contains(line, "level=INFO") {
			summary.InfoCount++
		} else if strings.Contains(line, "level=DEBUG") {
			summary.DebugCount++
		}

		// Extract latency metrics
		if idx := strings.Index(line, "duration_ms="); idx != -1 {
			latencyStr := extractNumericValue(line[idx+12:])
			if latency, err := strconv.ParseFloat(latencyStr, 64); err == nil {
				totalLatency += latency
				latencyCount++
			}
		}
	}

	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("error reading log file: %w", err)
	}

	// Calculate averages and rates
	if latencyCount > 0 {
		summary.AvgLatencyMs = totalLatency / float64(latencyCount)
	}
	if summary.TotalEntries > 0 {
		summary.ErrorRate = float64(summary.ErrorCount) / float64(summary.TotalEntries) * 100
	}

	return summary, nil
}

// extractQuotedValue extracts a value between quotes or until a space
func extractQuotedValue(s string) string {
	if idx := strings.Index(s, "\""); idx != -1 {
		s = s[idx+1:]
		if endIdx := strings.Index(s, "\""); endIdx != -1 {
			return s[:endIdx]
		}
	}
	// Fallback: extract until space
	if idx := strings.Index(s, " "); idx != -1 {
		return s[:idx]
	}
	return s
}

// extractNumericValue extracts numeric value until first non-numeric character
func extractNumericValue(s string) string {
	var result strings.Builder
	for _, ch := range s {
		if (ch >= '0' && ch <= '9') || ch == '.' || ch == '-' {
			result.WriteRune(ch)
		} else {
			break
		}
	}
	return result.String()
}

// ===== STRUCTURED LOGGING FORMATTERS =====

// FormatChessMove formats chess move logging
func FormatChessMove(from, to string, piece string, confidence float64) map[string]any {
	return map[string]any{
		"from":       from,
		"to":         to,
		"piece":      piece,
		"confidence": fmt.Sprintf("%.2f%%", confidence*100),
	}
}

// FormatModelMetrics formats model metrics for logging
func FormatModelMetrics(loss, accuracy, valLoss, valAccuracy float64) map[string]any {
	return map[string]any{
		"loss":         fmt.Sprintf("%.4f", loss),
		"accuracy":     fmt.Sprintf("%.2f%%", accuracy),
		"val_loss":     fmt.Sprintf("%.4f", valLoss),
		"val_accuracy": fmt.Sprintf("%.2f%%", valAccuracy),
	}
}

// FormatPerformance formats performance metrics for logging
func FormatPerformance(latencyMs, throughput float64, cacheHit bool) map[string]any {
	return map[string]any{
		"latency_ms": fmt.Sprintf("%.2f", latencyMs),
		"throughput": fmt.Sprintf("%.1f/s", throughput),
		"cache_hit":  cacheHit,
	}
}
