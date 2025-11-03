package main

import (
	"bytes"
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"net/http"
	_ "net/http/pprof"
	"os"
	"os/signal"
	"runtime"
	"strings"
	"syscall"
	"time"

	"github.com/thyrook/partner/internal/config"
	"github.com/thyrook/partner/internal/decision"
	"github.com/thyrook/partner/internal/model"
	"github.com/thyrook/partner/internal/storage"
	"github.com/thyrook/partner/internal/training"
	"github.com/thyrook/partner/internal/vision"

	"go.uber.org/zap"
	"go.uber.org/zap/zapcore"
)

const (
	TestVersion   = "1.0.0"
	TestName      = "System Integration Test"
	TestDuration  = 5 * time.Minute
	StatsInterval = 10 * time.Second
)

// SystemMetrics tracks comprehensive system performance
type SystemMetrics struct {
	// Timing metrics
	StartTime     time.Time
	EndTime       time.Time
	TotalDuration time.Duration

	// Vision metrics
	TotalCaptures      int64
	SuccessfulCaptures int64
	FailedCaptures     int64
	CaptureLatencySum  time.Duration
	MinCaptureLatency  time.Duration
	MaxCaptureLatency  time.Duration

	// Inference metrics
	TotalInferences  int64
	InferenceTimeSum time.Duration
	MinInferenceTime time.Duration
	MaxInferenceTime time.Duration

	// Decision metrics
	TotalDecisions    int64
	HighConfDecisions int64 // >= 70%
	MedConfDecisions  int64 // 40-70%
	LowConfDecisions  int64 // < 40%
	AvgConfidence     float64

	// Storage metrics
	SamplesStored int64
	StorageWrites int64
	StorageReads  int64
	StorageErrors int64

	// System metrics
	InitialMemoryMB float64
	PeakMemoryMB    float64
	FinalMemoryMB   float64
	AvgCPUPercent   float64
	GoroutineCount  int
	GCCount         uint32

	// Frame rate metrics
	FramesProcessed int64
	ActualFPS       float64
	TargetFPS       float64
	FrameDrops      int64
}

// SnapshotMetrics for periodic reporting
type SnapshotMetrics struct {
	Timestamp         time.Time
	ElapsedSeconds    float64
	CapturesPerSec    float64
	InferencesPerSec  float64
	AvgCaptureLatency time.Duration
	AvgInferenceTime  time.Duration
	MemoryMB          float64
	CPUPercent        float64
	Goroutines        int
	ConfidenceAvg     float64
}

// TestController orchestrates the integration test
type TestController struct {
	cfg    *config.Config
	logger *zap.Logger

	// Components
	capturer *vision.Capturer
	store    *storage.ObservationStore
	net      *model.ChessNet
	trainer  *training.Trainer
	engine   *decision.DecisionEngine

	// Metrics
	metrics   *SystemMetrics
	snapshots []SnapshotMetrics

	// Control
	ctx      context.Context
	cancel   context.CancelFunc
	doneChan chan struct{}

	// Profiling
	cpuSamples  []float64
	memSamples  []float64
	lastCPUTime time.Time
	lastCPUStat CPUStat
}

// CPUStat for CPU usage calculation
type CPUStat struct {
	UserTime   time.Duration
	SystemTime time.Duration
}

func NewTestController(cfg *config.Config, logger *zap.Logger) (*TestController, error) {
	ctx, cancel := context.WithCancel(context.Background())

	tc := &TestController{
		cfg:      cfg,
		logger:   logger,
		ctx:      ctx,
		cancel:   cancel,
		doneChan: make(chan struct{}),
		metrics: &SystemMetrics{
			MinCaptureLatency: time.Hour,
			MaxCaptureLatency: 0,
			MinInferenceTime:  time.Hour,
			MaxInferenceTime:  0,
			TargetFPS:         2.0, // 2 FPS for test
		},
		snapshots: make([]SnapshotMetrics, 0, 30), // ~30 snapshots for 5 min
	}

	return tc, nil
}

func (tc *TestController) Initialize() error {
	tc.logger.Info("Initializing system components...")

	// 1. Initialize vision system
	tc.logger.Info("Setting up vision capture system...")
	capturer := vision.NewCapturer(
		tc.cfg.Vision.ScreenRegion.X,
		tc.cfg.Vision.ScreenRegion.Y,
		tc.cfg.Vision.ScreenRegion.Width,
		tc.cfg.Vision.ScreenRegion.Height,
		tc.cfg.Vision.BoardSize,
		tc.cfg.Vision.DiffThreshold,
	)
	tc.capturer = capturer
	tc.logger.Info("Vision system ready", zap.Int("region_width", tc.cfg.Vision.ScreenRegion.Width))

	// 2. Initialize storage
	tc.logger.Info("Initializing observation storage...")
	dbPath := tc.cfg.Training.DBPath
	if dbPath == "" {
		dbPath = "data/observations.db"
	}
	store, err := storage.NewObservationStore(dbPath, tc.cfg.Training.ReplayBufferSize)
	if err != nil {
		return fmt.Errorf("failed to create storage: %w", err)
	}
	tc.store = store

	count, err := store.CountSamples()
	if err != nil {
		return fmt.Errorf("failed to get sample count: %w", err)
	}
	tc.logger.Info("Storage initialized",
		zap.Uint64("existing_samples", count),
		zap.Int("capacity", tc.cfg.Training.ReplayBufferSize))

	// 3. Initialize model
	tc.logger.Info("Loading neural network model...")
	modelPath := tc.cfg.Model.ModelPath
	if modelPath == "" {
		modelPath = "data/model.bin"
	}

	// Create model first
	net, err := model.NewChessNet(
		tc.cfg.Model.InputSize,
		tc.cfg.Model.HiddenSize,
		tc.cfg.Model.OutputSize,
	)
	if err != nil {
		return fmt.Errorf("failed to create model: %w", err)
	}

	// Try to load existing weights
	if _, statErr := os.Stat(modelPath); statErr == nil {
		if err := net.Load(modelPath); err != nil {
			tc.logger.Warn("Failed to load existing model weights, using fresh model", zap.Error(err))
		} else {
			tc.logger.Info("Existing model weights loaded", zap.String("path", modelPath))
		}
	} else {
		params := tc.cfg.Model.InputSize*tc.cfg.Model.HiddenSize + tc.cfg.Model.HiddenSize*tc.cfg.Model.OutputSize
		tc.logger.Info("New model created", zap.Int("approx_params", params))
	}
	tc.net = net

	// 4. Initialize trainer (skipped - uses old API, not needed for integration test)
	tc.logger.Info("Skipping trainer initialization (not used in integration test)")
	tc.trainer = nil

	// 5. Initialize decision engine
	tc.logger.Info("Setting up decision engine...")
	engine := decision.NewDecisionEngine(
		net,
		capturer,
		tc.cfg.Interface.ConfidenceThreshold,
		5, // top-5 moves
		tc.logger,
	)
	tc.engine = engine
	tc.logger.Info("Decision engine ready")

	// Record initial memory
	tc.metrics.InitialMemoryMB = tc.getMemoryMB()
	tc.logger.Info("System initialization complete",
		zap.Float64("initial_memory_mb", tc.metrics.InitialMemoryMB))

	return nil
}

func (tc *TestController) RunIntegrationTest(duration time.Duration) error {
	tc.logger.Info("Starting integration test",
		zap.Duration("duration", duration),
		zap.Float64("target_fps", tc.metrics.TargetFPS))

	tc.metrics.StartTime = time.Now()

	// Start profiling goroutine
	go tc.profileSystem()

	// Start decision collection goroutine
	go tc.collectDecisions(duration)

	// Wait for completion or interruption
	timer := time.NewTimer(duration)
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, os.Interrupt, syscall.SIGTERM)

	select {
	case <-timer.C:
		tc.logger.Info("Test duration completed")
	case <-sigChan:
		tc.logger.Info("Test interrupted by signal")
	case <-tc.doneChan:
		tc.logger.Info("Test completed early")
	}

	tc.cancel()
	tc.metrics.EndTime = time.Now()
	tc.metrics.TotalDuration = tc.metrics.EndTime.Sub(tc.metrics.StartTime)

	// Final memory snapshot
	tc.metrics.FinalMemoryMB = tc.getMemoryMB()
	tc.metrics.GoroutineCount = runtime.NumGoroutine()

	var memStats runtime.MemStats
	runtime.ReadMemStats(&memStats)
	tc.metrics.GCCount = memStats.NumGC

	return nil
}

func (tc *TestController) collectDecisions(duration time.Duration) {
	defer close(tc.doneChan)

	ticker := time.NewTicker(time.Second / 2) // 2 FPS
	defer ticker.Stop()

	statsTicker := time.NewTicker(StatsInterval)
	defer statsTicker.Stop()

	frameNumber := int64(0)
	lastFrameTime := time.Now()

	for {
		select {
		case <-tc.ctx.Done():
			return
		case <-ticker.C:
			frameNumber++
			frameStart := time.Now()

			// Check for frame drops
			timeSinceLastFrame := frameStart.Sub(lastFrameTime)
			expectedInterval := time.Second / 2
			if timeSinceLastFrame > expectedInterval*3/2 {
				tc.metrics.FrameDrops++
			}
			lastFrameTime = frameStart

			// Execute decision cycle
			tc.executeDecisionCycle(frameNumber)

			// Calculate actual FPS
			tc.metrics.FramesProcessed = frameNumber
			elapsed := time.Since(tc.metrics.StartTime).Seconds()
			tc.metrics.ActualFPS = float64(frameNumber) / elapsed

		case <-statsTicker.C:
			tc.captureSnapshot()
		}
	}
}

func (tc *TestController) executeDecisionCycle(frameNum int64) {
	cycleStart := time.Now()

	// 1. Capture board
	captureStart := time.Now()
	mat, err := tc.capturer.CaptureFrame()
	captureLatency := time.Since(captureStart)

	tc.metrics.TotalCaptures++
	tc.metrics.CaptureLatencySum += captureLatency

	if captureLatency < tc.metrics.MinCaptureLatency {
		tc.metrics.MinCaptureLatency = captureLatency
	}
	if captureLatency > tc.metrics.MaxCaptureLatency {
		tc.metrics.MaxCaptureLatency = captureLatency
	}

	if err != nil {
		tc.metrics.FailedCaptures++
		tc.logger.Warn("Capture failed",
			zap.Int64("frame", frameNum),
			zap.Error(err))
		return
	}
	tc.metrics.SuccessfulCaptures++

	// Convert frame to board state (simplified: flatten mat data)
	boardState := make([]float64, 64)
	if mat != nil {
		// Simple conversion: take grayscale values
		for i := 0; i < 64 && i < mat.Rows()*mat.Cols(); i++ {
			boardState[i] = float64(i%256) / 255.0 // Placeholder
		}
		mat.Close()
	}

	// 2. Run inference
	inferenceStart := time.Now()
	predictions, err := tc.net.Predict(boardState)
	inferenceTime := time.Since(inferenceStart)

	if err != nil {
		tc.logger.Warn("Inference failed",
			zap.Int64("frame", frameNum),
			zap.Error(err))
		return
	}

	tc.metrics.TotalInferences++
	tc.metrics.InferenceTimeSum += inferenceTime

	if inferenceTime < tc.metrics.MinInferenceTime {
		tc.metrics.MinInferenceTime = inferenceTime
	}
	if inferenceTime > tc.metrics.MaxInferenceTime {
		tc.metrics.MaxInferenceTime = inferenceTime
	}

	// 3. Process decision
	maxConf := 0.0
	for _, pred := range predictions {
		if pred > maxConf {
			maxConf = pred
		}
	}

	tc.metrics.TotalDecisions++
	tc.metrics.AvgConfidence = (tc.metrics.AvgConfidence*float64(tc.metrics.TotalDecisions-1) + maxConf) / float64(tc.metrics.TotalDecisions)

	if maxConf >= 0.70 {
		tc.metrics.HighConfDecisions++
	} else if maxConf >= 0.40 {
		tc.metrics.MedConfDecisions++
	} else {
		tc.metrics.LowConfDecisions++
	}

	// 4. Store observation (every 5th frame)
	if frameNum%5 == 0 {
		// Get highest confidence move as label
		maxIdx := 0
		for i, pred := range predictions {
			if pred > predictions[maxIdx] {
				maxIdx = i
			}
		}

		if err := tc.store.StoreSample(boardState, maxIdx); err != nil {
			tc.metrics.StorageErrors++
			tc.logger.Warn("Storage failed", zap.Error(err))
		} else {
			tc.metrics.SamplesStored++
			tc.metrics.StorageWrites++
		}
	}

	cycleDuration := time.Since(cycleStart)

	// Log detailed cycle info every 20 frames
	if frameNum%20 == 0 {
		tc.logger.Info("Decision cycle",
			zap.Int64("frame", frameNum),
			zap.Duration("total", cycleDuration),
			zap.Duration("capture", captureLatency),
			zap.Duration("inference", inferenceTime),
			zap.Float64("confidence", maxConf),
			zap.Float64("fps", tc.metrics.ActualFPS))
	}
}

func (tc *TestController) profileSystem() {
	ticker := time.NewTicker(time.Second)
	defer ticker.Stop()

	tc.lastCPUTime = time.Now()
	tc.lastCPUStat = tc.getCPUStat()

	for {
		select {
		case <-tc.ctx.Done():
			return
		case <-ticker.C:
			// Sample memory
			memMB := tc.getMemoryMB()
			tc.memSamples = append(tc.memSamples, memMB)

			if memMB > tc.metrics.PeakMemoryMB {
				tc.metrics.PeakMemoryMB = memMB
			}

			// Sample CPU
			cpuPercent := tc.getCPUPercent()
			tc.cpuSamples = append(tc.cpuSamples, cpuPercent)

			// Calculate average CPU
			if len(tc.cpuSamples) > 0 {
				sum := 0.0
				for _, v := range tc.cpuSamples {
					sum += v
				}
				tc.metrics.AvgCPUPercent = sum / float64(len(tc.cpuSamples))
			}
		}
	}
}

func (tc *TestController) captureSnapshot() {
	elapsed := time.Since(tc.metrics.StartTime).Seconds()

	snapshot := SnapshotMetrics{
		Timestamp:      time.Now(),
		ElapsedSeconds: elapsed,
		MemoryMB:       tc.getMemoryMB(),
		CPUPercent:     tc.getCPUPercent(),
		Goroutines:     runtime.NumGoroutine(),
		ConfidenceAvg:  tc.metrics.AvgConfidence,
	}

	if elapsed > 0 {
		snapshot.CapturesPerSec = float64(tc.metrics.TotalCaptures) / elapsed
		snapshot.InferencesPerSec = float64(tc.metrics.TotalInferences) / elapsed
	}

	if tc.metrics.TotalCaptures > 0 {
		snapshot.AvgCaptureLatency = tc.metrics.CaptureLatencySum / time.Duration(tc.metrics.TotalCaptures)
	}

	if tc.metrics.TotalInferences > 0 {
		snapshot.AvgInferenceTime = tc.metrics.InferenceTimeSum / time.Duration(tc.metrics.TotalInferences)
	}

	tc.snapshots = append(tc.snapshots, snapshot)

	tc.logger.Info("Performance snapshot",
		zap.Float64("elapsed_sec", elapsed),
		zap.Float64("captures_per_sec", snapshot.CapturesPerSec),
		zap.Float64("inferences_per_sec", snapshot.InferencesPerSec),
		zap.Duration("avg_capture", snapshot.AvgCaptureLatency),
		zap.Duration("avg_inference", snapshot.AvgInferenceTime),
		zap.Float64("memory_mb", snapshot.MemoryMB),
		zap.Float64("cpu_percent", snapshot.CPUPercent),
		zap.Int("goroutines", snapshot.Goroutines),
		zap.Float64("confidence", snapshot.ConfidenceAvg))
}

func (tc *TestController) getMemoryMB() float64 {
	var m runtime.MemStats
	runtime.ReadMemStats(&m)
	return float64(m.Alloc) / 1024.0 / 1024.0
}

func (tc *TestController) getCPUStat() CPUStat {
	var rusage syscall.Rusage
	syscall.Getrusage(syscall.RUSAGE_SELF, &rusage)

	return CPUStat{
		UserTime:   time.Duration(rusage.Utime.Nano()),
		SystemTime: time.Duration(rusage.Stime.Nano()),
	}
}

func (tc *TestController) getCPUPercent() float64 {
	now := time.Now()
	currentStat := tc.getCPUStat()

	wallTime := now.Sub(tc.lastCPUTime)
	cpuTime := (currentStat.UserTime + currentStat.SystemTime) -
		(tc.lastCPUStat.UserTime + tc.lastCPUStat.SystemTime)

	tc.lastCPUTime = now
	tc.lastCPUStat = currentStat

	if wallTime == 0 {
		return 0
	}

	// Single core percentage
	return (float64(cpuTime) / float64(wallTime)) * 100.0
}

func (tc *TestController) GenerateReport() string {
	report := &Report{
		TestInfo: TestInfo{
			Name:      TestName,
			Version:   TestVersion,
			StartTime: tc.metrics.StartTime,
			EndTime:   tc.metrics.EndTime,
			Duration:  tc.metrics.TotalDuration,
		},
		Metrics:   tc.metrics,
		Snapshots: tc.snapshots,
	}

	return report.Format()
}

func (tc *TestController) Cleanup() {
	tc.logger.Info("Cleaning up system resources...")

	if tc.store != nil {
		tc.store.Close()
	}

	tc.logger.Info("Cleanup complete")
}

// Report structures
type TestInfo struct {
	Name      string
	Version   string
	StartTime time.Time
	EndTime   time.Time
	Duration  time.Duration
}

type Report struct {
	TestInfo  TestInfo
	Metrics   *SystemMetrics
	Snapshots []SnapshotMetrics
}

func (r *Report) Format() string {
	var b []byte
	buf := &bytes.Buffer{}

	// ASCII box drawing
	line := "═══════════════════════════════════════════════════════════════════════"

	fmt.Fprintf(buf, "\n")
	fmt.Fprintf(buf, "╔%s╗\n", line)
	fmt.Fprintf(buf, "║  P.A.R.T.N.E.R SYSTEM INTEGRATION TEST REPORT%s║\n",
		strings.Repeat(" ", len(line)-48))
	fmt.Fprintf(buf, "╚%s╝\n", line)
	fmt.Fprintf(buf, "\n")

	// Test Information
	fmt.Fprintf(buf, "📋 TEST INFORMATION\n")
	fmt.Fprintf(buf, "─────────────────────────────────────────────────────────────────────\n")
	fmt.Fprintf(buf, "Test Name:     %s\n", r.TestInfo.Name)
	fmt.Fprintf(buf, "Version:       %s\n", r.TestInfo.Version)
	fmt.Fprintf(buf, "Start Time:    %s\n", r.TestInfo.StartTime.Format("2006-01-02 15:04:05"))
	fmt.Fprintf(buf, "End Time:      %s\n", r.TestInfo.EndTime.Format("2006-01-02 15:04:05"))
	fmt.Fprintf(buf, "Duration:      %s\n", r.TestInfo.Duration.Round(time.Millisecond))
	fmt.Fprintf(buf, "\n")

	// Vision System Performance
	fmt.Fprintf(buf, "👁️  VISION SYSTEM PERFORMANCE\n")
	fmt.Fprintf(buf, "─────────────────────────────────────────────────────────────────────\n")
	fmt.Fprintf(buf, "Total Captures:      %d\n", r.Metrics.TotalCaptures)
	fmt.Fprintf(buf, "Successful:          %d (%.1f%%)\n",
		r.Metrics.SuccessfulCaptures,
		float64(r.Metrics.SuccessfulCaptures)/float64(r.Metrics.TotalCaptures)*100)
	fmt.Fprintf(buf, "Failed:              %d (%.1f%%)\n",
		r.Metrics.FailedCaptures,
		float64(r.Metrics.FailedCaptures)/float64(r.Metrics.TotalCaptures)*100)

	avgCapture := time.Duration(0)
	if r.Metrics.TotalCaptures > 0 {
		avgCapture = r.Metrics.CaptureLatencySum / time.Duration(r.Metrics.TotalCaptures)
	}

	fmt.Fprintf(buf, "Avg Latency:         %s\n", avgCapture.Round(time.Microsecond))
	fmt.Fprintf(buf, "Min Latency:         %s\n", r.Metrics.MinCaptureLatency.Round(time.Microsecond))
	fmt.Fprintf(buf, "Max Latency:         %s\n", r.Metrics.MaxCaptureLatency.Round(time.Microsecond))
	fmt.Fprintf(buf, "\n")

	// Inference Performance
	fmt.Fprintf(buf, "🧠 INFERENCE PERFORMANCE\n")
	fmt.Fprintf(buf, "─────────────────────────────────────────────────────────────────────\n")
	fmt.Fprintf(buf, "Total Inferences:    %d\n", r.Metrics.TotalInferences)

	avgInference := time.Duration(0)
	if r.Metrics.TotalInferences > 0 {
		avgInference = r.Metrics.InferenceTimeSum / time.Duration(r.Metrics.TotalInferences)
	}

	fmt.Fprintf(buf, "Avg Time:            %s\n", avgInference.Round(time.Microsecond))
	fmt.Fprintf(buf, "Min Time:            %s\n", r.Metrics.MinInferenceTime.Round(time.Microsecond))
	fmt.Fprintf(buf, "Max Time:            %s\n", r.Metrics.MaxInferenceTime.Round(time.Microsecond))

	if avgInference > 0 {
		inferenceRate := float64(time.Second) / float64(avgInference)
		fmt.Fprintf(buf, "Inferences/sec:      %.1f\n", inferenceRate)
	}
	fmt.Fprintf(buf, "\n")

	// Decision Quality
	fmt.Fprintf(buf, "🎯 DECISION QUALITY\n")
	fmt.Fprintf(buf, "─────────────────────────────────────────────────────────────────────\n")
	fmt.Fprintf(buf, "Total Decisions:     %d\n", r.Metrics.TotalDecisions)
	fmt.Fprintf(buf, "High Confidence:     %d (%.1f%%) [≥70%%]\n",
		r.Metrics.HighConfDecisions,
		float64(r.Metrics.HighConfDecisions)/float64(r.Metrics.TotalDecisions)*100)
	fmt.Fprintf(buf, "Med Confidence:      %d (%.1f%%) [40-70%%]\n",
		r.Metrics.MedConfDecisions,
		float64(r.Metrics.MedConfDecisions)/float64(r.Metrics.TotalDecisions)*100)
	fmt.Fprintf(buf, "Low Confidence:      %d (%.1f%%) [<40%%]\n",
		r.Metrics.LowConfDecisions,
		float64(r.Metrics.LowConfDecisions)/float64(r.Metrics.TotalDecisions)*100)
	fmt.Fprintf(buf, "Avg Confidence:      %.2f%%\n", r.Metrics.AvgConfidence*100)
	fmt.Fprintf(buf, "\n")

	// Storage Performance
	fmt.Fprintf(buf, "💾 STORAGE PERFORMANCE\n")
	fmt.Fprintf(buf, "─────────────────────────────────────────────────────────────────────\n")
	fmt.Fprintf(buf, "Samples Stored:      %d\n", r.Metrics.SamplesStored)
	fmt.Fprintf(buf, "Write Operations:    %d\n", r.Metrics.StorageWrites)
	fmt.Fprintf(buf, "Read Operations:     %d\n", r.Metrics.StorageReads)
	fmt.Fprintf(buf, "Errors:              %d\n", r.Metrics.StorageErrors)
	if r.Metrics.StorageWrites > 0 {
		successRate := float64(r.Metrics.StorageWrites-r.Metrics.StorageErrors) / float64(r.Metrics.StorageWrites) * 100
		fmt.Fprintf(buf, "Success Rate:        %.2f%%\n", successRate)
	}
	fmt.Fprintf(buf, "\n")

	// Frame Rate Analysis
	fmt.Fprintf(buf, "📊 FRAME RATE ANALYSIS\n")
	fmt.Fprintf(buf, "─────────────────────────────────────────────────────────────────────\n")
	fmt.Fprintf(buf, "Frames Processed:    %d\n", r.Metrics.FramesProcessed)
	fmt.Fprintf(buf, "Target FPS:          %.2f\n", r.Metrics.TargetFPS)
	fmt.Fprintf(buf, "Actual FPS:          %.2f\n", r.Metrics.ActualFPS)
	fmt.Fprintf(buf, "Frame Drops:         %d\n", r.Metrics.FrameDrops)

	fpsEfficiency := (r.Metrics.ActualFPS / r.Metrics.TargetFPS) * 100
	if fpsEfficiency > 100 {
		fpsEfficiency = 100
	}
	fmt.Fprintf(buf, "FPS Efficiency:      %.1f%%\n", fpsEfficiency)
	fmt.Fprintf(buf, "\n")

	// System Resource Usage
	fmt.Fprintf(buf, "⚙️  SYSTEM RESOURCE USAGE\n")
	fmt.Fprintf(buf, "─────────────────────────────────────────────────────────────────────\n")
	fmt.Fprintf(buf, "Initial Memory:      %.2f MB\n", r.Metrics.InitialMemoryMB)
	fmt.Fprintf(buf, "Peak Memory:         %.2f MB\n", r.Metrics.PeakMemoryMB)
	fmt.Fprintf(buf, "Final Memory:        %.2f MB\n", r.Metrics.FinalMemoryMB)
	fmt.Fprintf(buf, "Memory Growth:       %.2f MB (%.1f%%)\n",
		r.Metrics.FinalMemoryMB-r.Metrics.InitialMemoryMB,
		(r.Metrics.FinalMemoryMB-r.Metrics.InitialMemoryMB)/r.Metrics.InitialMemoryMB*100)
	fmt.Fprintf(buf, "Avg CPU Usage:       %.2f%%\n", r.Metrics.AvgCPUPercent)
	fmt.Fprintf(buf, "Goroutines:          %d\n", r.Metrics.GoroutineCount)
	fmt.Fprintf(buf, "GC Collections:      %d\n", r.Metrics.GCCount)
	fmt.Fprintf(buf, "\n")

	// Performance Snapshots Summary
	if len(r.Snapshots) > 0 {
		fmt.Fprintf(buf, "📸 PERFORMANCE SNAPSHOTS (%d samples)\n", len(r.Snapshots))
		fmt.Fprintf(buf, "─────────────────────────────────────────────────────────────────────\n")
		fmt.Fprintf(buf, "Time(s) | Cap/s | Inf/s | CapLat    | InfTime   | Mem(MB) | CPU%% | Conf%%\n")
		fmt.Fprintf(buf, "────────┼───────┼───────┼───────────┼───────────┼─────────┼──────┼───────\n")

		// Show first 3, middle 3, last 3
		printSnapshot := func(s SnapshotMetrics) {
			fmt.Fprintf(buf, "%7.0f │ %5.1f │ %5.1f │ %9s │ %9s │ %7.1f │ %4.1f │ %5.1f\n",
				s.ElapsedSeconds,
				s.CapturesPerSec,
				s.InferencesPerSec,
				s.AvgCaptureLatency.Round(time.Microsecond).String(),
				s.AvgInferenceTime.Round(time.Microsecond).String(),
				s.MemoryMB,
				s.CPUPercent,
				s.ConfidenceAvg*100)
		}

		count := len(r.Snapshots)
		if count <= 9 {
			for _, s := range r.Snapshots {
				printSnapshot(s)
			}
		} else {
			for i := 0; i < 3; i++ {
				printSnapshot(r.Snapshots[i])
			}
			fmt.Fprintf(buf, "   ...  │  ...  │  ...  │    ...    │    ...    │   ...   │ ...  │  ... \n")
			mid := count / 2
			for i := mid - 1; i <= mid+1; i++ {
				printSnapshot(r.Snapshots[i])
			}
			fmt.Fprintf(buf, "   ...  │  ...  │  ...  │    ...    │    ...    │   ...   │ ...  │  ... \n")
			for i := count - 3; i < count; i++ {
				printSnapshot(r.Snapshots[i])
			}
		}
		fmt.Fprintf(buf, "\n")
	}

	// Bottleneck Analysis
	fmt.Fprintf(buf, "🔍 BOTTLENECK ANALYSIS\n")
	fmt.Fprintf(buf, "─────────────────────────────────────────────────────────────────────\n")

	totalTime := float64(avgCapture + avgInference)
	capturePercent := float64(avgCapture) / totalTime * 100
	inferencePercent := float64(avgInference) / totalTime * 100

	fmt.Fprintf(buf, "Capture Time:        %.1f%% of cycle\n", capturePercent)
	fmt.Fprintf(buf, "Inference Time:      %.1f%% of cycle\n", inferencePercent)

	if capturePercent > inferencePercent {
		fmt.Fprintf(buf, "⚠️  Primary Bottleneck: VISION CAPTURE\n")
	} else {
		fmt.Fprintf(buf, "⚠️  Primary Bottleneck: INFERENCE\n")
	}
	fmt.Fprintf(buf, "\n")

	// Optimization Recommendations
	fmt.Fprintf(buf, "💡 OPTIMIZATION RECOMMENDATIONS\n")
	fmt.Fprintf(buf, "─────────────────────────────────────────────────────────────────────\n")

	recommendations := r.generateRecommendations()
	for i, rec := range recommendations {
		fmt.Fprintf(buf, "%d. %s\n", i+1, rec)
	}
	fmt.Fprintf(buf, "\n")

	// Summary
	fmt.Fprintf(buf, "✅ TEST SUMMARY\n")
	fmt.Fprintf(buf, "─────────────────────────────────────────────────────────────────────\n")

	captureSuccess := float64(r.Metrics.SuccessfulCaptures) / float64(r.Metrics.TotalCaptures) * 100
	storageSuccess := 100.0
	if r.Metrics.StorageWrites > 0 {
		storageSuccess = float64(r.Metrics.StorageWrites-r.Metrics.StorageErrors) / float64(r.Metrics.StorageWrites) * 100
	}

	fmt.Fprintf(buf, "Overall Status:      ")
	if captureSuccess > 95 && storageSuccess > 95 && r.Metrics.ActualFPS > r.Metrics.TargetFPS*0.9 {
		fmt.Fprintf(buf, "✅ EXCELLENT\n")
	} else if captureSuccess > 85 && storageSuccess > 85 {
		fmt.Fprintf(buf, "✓ GOOD\n")
	} else {
		fmt.Fprintf(buf, "⚠️  NEEDS ATTENTION\n")
	}

	fmt.Fprintf(buf, "Capture Reliability: %.1f%%\n", captureSuccess)
	fmt.Fprintf(buf, "Storage Reliability: %.1f%%\n", storageSuccess)
	fmt.Fprintf(buf, "FPS Achievement:     %.1f%%\n", fpsEfficiency)
	fmt.Fprintf(buf, "\n")

	fmt.Fprintf(buf, "╔%s╗\n", line)
	fmt.Fprintf(buf, "║  END OF REPORT%s║\n", strings.Repeat(" ", len(line)-17))
	fmt.Fprintf(buf, "╚%s╝\n", line)

	b = buf.Bytes()
	return string(b)
}

func (r *Report) generateRecommendations() []string {
	var recs []string

	m := r.Metrics

	// Memory recommendations
	memGrowth := m.FinalMemoryMB - m.InitialMemoryMB
	if memGrowth > 10 {
		recs = append(recs, fmt.Sprintf(
			"MEMORY: Detected %.1f MB growth. Consider tensor reuse and object pooling.",
			memGrowth))
	}

	if m.PeakMemoryMB > 100 {
		recs = append(recs, fmt.Sprintf(
			"MEMORY: Peak usage %.1f MB is high. Profile with pprof heap analysis.",
			m.PeakMemoryMB))
	}

	// CPU recommendations
	if m.AvgCPUPercent > 70 {
		recs = append(recs, fmt.Sprintf(
			"CPU: Average %.1f%% usage is high. Consider async processing or batch optimization.",
			m.AvgCPUPercent))
	}

	// Capture recommendations
	avgCapture := time.Duration(0)
	if m.TotalCaptures > 0 {
		avgCapture = m.CaptureLatencySum / time.Duration(m.TotalCaptures)
	}

	if avgCapture > 50*time.Millisecond {
		recs = append(recs, fmt.Sprintf(
			"VISION: Capture latency %s is high. Use async capture with frame buffer.",
			avgCapture.Round(time.Millisecond)))
	}

	captureSuccess := float64(m.SuccessfulCaptures) / float64(m.TotalCaptures) * 100
	if captureSuccess < 95 {
		recs = append(recs, fmt.Sprintf(
			"VISION: Capture success %.1f%% is low. Add retry logic and error recovery.",
			captureSuccess))
	}

	// Inference recommendations
	avgInference := time.Duration(0)
	if m.TotalInferences > 0 {
		avgInference = m.InferenceTimeSum / time.Duration(m.TotalInferences)
	}

	if avgInference > 10*time.Millisecond {
		recs = append(recs, fmt.Sprintf(
			"INFERENCE: Time %s is high. Consider model quantization or pruning.",
			avgInference.Round(time.Millisecond)))
	}

	// FPS recommendations
	if m.FrameDrops > m.FramesProcessed/10 {
		recs = append(recs, fmt.Sprintf(
			"FPS: %d frame drops detected. Reduce processing frequency or optimize pipeline.",
			m.FrameDrops))
	}

	if m.ActualFPS < m.TargetFPS*0.9 {
		recs = append(recs, fmt.Sprintf(
			"FPS: Actual %.2f < target %.2f. Optimize critical path or reduce target.",
			m.ActualFPS, m.TargetFPS))
	}

	// Confidence recommendations
	if m.AvgConfidence < 0.5 {
		recs = append(recs, fmt.Sprintf(
			"MODEL: Average confidence %.1f%% is low. Collect more training data.",
			m.AvgConfidence*100))
	}

	if m.LowConfDecisions > m.TotalDecisions/2 {
		recs = append(recs, fmt.Sprintf(
			"MODEL: %d low-confidence decisions. Model needs training with real data.",
			m.LowConfDecisions))
	}

	// Storage recommendations
	if m.StorageErrors > 0 {
		recs = append(recs, fmt.Sprintf(
			"STORAGE: %d errors detected. Check disk space and database integrity.",
			m.StorageErrors))
	}

	// GC recommendations
	duration := r.TestInfo.Duration.Seconds()
	if duration > 0 {
		gcRate := float64(m.GCCount) / duration
		if gcRate > 2 {
			recs = append(recs, fmt.Sprintf(
				"GC: %.1f collections/sec is high. Reduce allocations in hot path.",
				gcRate))
		}
	}

	// Pipeline optimization
	totalTime := float64(avgCapture + avgInference)
	if totalTime > 0 {
		capturePercent := float64(avgCapture) / totalTime * 100
		if capturePercent > 70 {
			recs = append(recs,
				"PIPELINE: Capture dominates cycle. Implement double-buffering or async capture.")
		} else if capturePercent < 30 {
			recs = append(recs,
				"PIPELINE: Inference dominates cycle. Consider model optimization or GPU acceleration.")
		}
	}

	// Goroutine recommendations
	if m.GoroutineCount > 50 {
		recs = append(recs, fmt.Sprintf(
			"CONCURRENCY: %d goroutines is high. Check for goroutine leaks.",
			m.GoroutineCount))
	}

	// General recommendations
	if len(recs) == 0 {
		recs = append(recs, "✅ System performing well. Consider stress testing with longer duration.")
		recs = append(recs, "💾 Implement tensor pooling to reduce GC pressure during inference.")
		recs = append(recs, "⚡ Add batching for storage writes to improve throughput.")
		recs = append(recs, "🎯 Consider GPU acceleration if targeting >10 FPS real-time processing.")
	}

	return recs
}

// Export report to JSON
func (tc *TestController) ExportJSON(path string) error {
	report := map[string]interface{}{
		"test_info": map[string]interface{}{
			"name":       TestName,
			"version":    TestVersion,
			"start_time": tc.metrics.StartTime,
			"end_time":   tc.metrics.EndTime,
			"duration":   tc.metrics.TotalDuration.String(),
		},
		"metrics":   tc.metrics,
		"snapshots": tc.snapshots,
	}

	data, err := json.MarshalIndent(report, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to marshal JSON: %w", err)
	}

	if err := os.WriteFile(path, data, 0644); err != nil {
		return fmt.Errorf("failed to write JSON: %w", err)
	}

	tc.logger.Info("Report exported to JSON", zap.String("path", path))
	return nil
}

func main() {
	// Command-line flags
	configPath := flag.String("config", "config.json", "Path to configuration file")
	duration := flag.Duration("duration", TestDuration, "Test duration")
	profile := flag.Bool("profile", true, "Enable pprof profiling")
	exportJSON := flag.String("export", "", "Export report to JSON file")
	verbose := flag.Bool("verbose", false, "Enable verbose logging")
	flag.Parse()

	// Setup logger
	logConfig := zap.NewProductionConfig()
	if *verbose {
		logConfig.Level = zap.NewAtomicLevelAt(zapcore.DebugLevel)
	}
	logConfig.EncoderConfig.EncodeTime = zapcore.ISO8601TimeEncoder

	logger, err := logConfig.Build()
	if err != nil {
		log.Fatalf("Failed to create logger: %v", err)
	}
	defer logger.Sync()

	// Start profiling server
	if *profile {
		go func() {
			logger.Info("Starting pprof server on :6060")
			if err := http.ListenAndServe(":6060", nil); err != nil {
				logger.Error("pprof server failed", zap.Error(err))
			}
		}()
	}

	// Load configuration
	cfg, err := config.Load(*configPath)
	if err != nil {
		logger.Fatal("Failed to load config", zap.Error(err))
	}

	// Create test controller
	tc, err := NewTestController(cfg, logger)
	if err != nil {
		logger.Fatal("Failed to create test controller", zap.Error(err))
	}

	// Initialize system
	logger.Info("=== P.A.R.T.N.E.R System Integration Test ===")
	logger.Info("Initializing system components...")

	if err := tc.Initialize(); err != nil {
		logger.Fatal("System initialization failed", zap.Error(err))
	}

	// Run integration test
	logger.Info("Starting integration test", zap.Duration("duration", *duration))

	if err := tc.RunIntegrationTest(*duration); err != nil {
		logger.Error("Integration test failed", zap.Error(err))
	}

	// Generate report
	logger.Info("Generating test report...")
	report := tc.GenerateReport()

	// Print report to console
	fmt.Println(report)

	// Export to JSON if requested
	if *exportJSON != "" {
		if err := tc.ExportJSON(*exportJSON); err != nil {
			logger.Error("Failed to export JSON", zap.Error(err))
		}
	}

	// Cleanup
	tc.Cleanup()

	logger.Info("Integration test complete")
}
