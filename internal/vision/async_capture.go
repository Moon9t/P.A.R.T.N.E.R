package vision

import (
	"context"
	"fmt"
	"image"
	"sync"
	"sync/atomic"
	"time"

	"github.com/kbinani/screenshot"
	"gocv.io/x/gocv"
)

// AsyncCapturer implements double-buffered asynchronous screen capture
// This dramatically improves performance by allowing capture and inference to run in parallel
type AsyncCapturer struct {
	// Configuration
	region        image.Rectangle
	boardSize     int
	diffThreshold float64
	targetFPS     int

	// Double buffering
	bufferA     *gocv.Mat
	bufferB     *gocv.Mat
	activeIdx   int32 // 0 for A, 1 for B
	bufferMu    sync.RWMutex
	bufferReady atomic.Bool

	// Capture metrics
	captureCount   atomic.Uint64
	captureErrors  atomic.Uint64
	lastCaptureMs  atomic.Int64
	totalCaptureMs atomic.Uint64

	// Background capture control
	ctx       context.Context
	cancel    context.CancelFunc
	wg        sync.WaitGroup
	frameChan chan captureResult
	isRunning atomic.Bool

	// Change detection
	lastFrame   *gocv.Mat
	lastFrameMu sync.Mutex
}

// captureResult holds the result of a capture operation
type captureResult struct {
	mat       gocv.Mat
	timestamp time.Time
	err       error
}

// AsyncCapturerConfig holds configuration for async capturer
type AsyncCapturerConfig struct {
	X             int
	Y             int
	Width         int
	Height        int
	BoardSize     int
	DiffThreshold float64
	TargetFPS     int
	BufferSize    int // Channel buffer size
}

// NewAsyncCapturer creates a new async capturer with double-buffering
func NewAsyncCapturer(cfg AsyncCapturerConfig) *AsyncCapturer {
	ctx, cancel := context.WithCancel(context.Background())

	if cfg.TargetFPS <= 0 {
		cfg.TargetFPS = 10 // Default to 10 FPS
	}
	if cfg.BufferSize <= 0 {
		cfg.BufferSize = 2 // Small buffer, we want latest frames
	}

	ac := &AsyncCapturer{
		region:        image.Rect(cfg.X, cfg.Y, cfg.X+cfg.Width, cfg.Y+cfg.Height),
		boardSize:     cfg.BoardSize,
		diffThreshold: cfg.DiffThreshold,
		targetFPS:     cfg.TargetFPS,
		ctx:           ctx,
		cancel:        cancel,
		frameChan:     make(chan captureResult, cfg.BufferSize),
	}

	// Initialize buffers
	ac.bufferA = &gocv.Mat{}
	ac.bufferB = &gocv.Mat{}

	return ac
}

// Start begins asynchronous capture in the background
func (ac *AsyncCapturer) Start() error {
	if ac.isRunning.Load() {
		return fmt.Errorf("async capturer already running")
	}

	ac.isRunning.Store(true)

	// Start capture goroutine
	ac.wg.Add(1)
	go ac.captureLoop()

	// Start buffer update goroutine
	ac.wg.Add(1)
	go ac.updateBufferLoop()

	return nil
}

// Stop stops the async capture
func (ac *AsyncCapturer) Stop() error {
	if !ac.isRunning.Load() {
		return nil
	}

	ac.isRunning.Store(false)
	ac.cancel()

	// Wait for goroutines to finish
	done := make(chan struct{})
	go func() {
		ac.wg.Wait()
		close(done)
	}()

	// Wait with timeout
	select {
	case <-done:
		// Clean shutdown
	case <-time.After(2 * time.Second):
		// Force shutdown after timeout
	}

	ac.cleanup()
	return nil
}

// captureLoop continuously captures frames in the background
func (ac *AsyncCapturer) captureLoop() {
	defer ac.wg.Done()

	interval := time.Second / time.Duration(ac.targetFPS)
	ticker := time.NewTicker(interval)
	defer ticker.Stop()

	for {
		select {
		case <-ac.ctx.Done():
			return
		case <-ticker.C:
			start := time.Now()
			mat, err := ac.captureScreen()
			duration := time.Since(start)

			if err != nil {
				ac.captureErrors.Add(1)
				// Send error result
				select {
				case ac.frameChan <- captureResult{mat: gocv.Mat{}, timestamp: time.Now(), err: err}:
				case <-ac.ctx.Done():
					return
				default:
					// Drop if channel full
				}
				continue
			}

			ac.captureCount.Add(1)
			ac.lastCaptureMs.Store(duration.Milliseconds())
			ac.totalCaptureMs.Add(uint64(duration.Microseconds()))

			// Send successful capture
			select {
			case ac.frameChan <- captureResult{mat: *mat, timestamp: time.Now(), err: nil}:
			case <-ac.ctx.Done():
				if mat != nil {
					mat.Close()
				}
				return
			default:
				// Drop old frame if channel full (we want latest)
				if mat != nil {
					mat.Close()
				}
			}
		}
	}
}

// updateBufferLoop receives captured frames and updates the double buffer
func (ac *AsyncCapturer) updateBufferLoop() {
	defer ac.wg.Done()

	for {
		select {
		case <-ac.ctx.Done():
			return
		case result := <-ac.frameChan:
			if result.err != nil {
				continue
			}
			if result.mat.Empty() {
				continue
			}

			ac.updateBuffer(&result.mat)
			result.mat.Close() // Close after copying
		}
	}
}

// captureScreen performs the actual screen capture
func (ac *AsyncCapturer) captureScreen() (*gocv.Mat, error) {
	// Capture screen region
	img, err := screenshot.CaptureRect(ac.region)
	if err != nil {
		return nil, fmt.Errorf("failed to capture screen: %w", err)
	}

	// Convert to OpenCV Mat
	mat, err := ac.imageToMat(img)
	if err != nil {
		return nil, fmt.Errorf("failed to convert image to mat: %w", err)
	}

	return mat, nil
}

// updateBuffer updates the inactive buffer and swaps
func (ac *AsyncCapturer) updateBuffer(newMat *gocv.Mat) {
	ac.bufferMu.Lock()
	defer ac.bufferMu.Unlock()

	// Get inactive buffer
	var targetBuffer *gocv.Mat
	currentIdx := ac.activeIdx

	if currentIdx == 0 {
		targetBuffer = ac.bufferB
	} else {
		targetBuffer = ac.bufferA
	}

	// Copy new frame to inactive buffer
	// Check if target buffer has been initialized (non-nil internal pointer)
	// gocv.Mat has an internal pointer 'p' - if it's nil (which happens with &gocv.Mat{}),
	// we need to create a new Mat instead of trying to check if it's empty
	if targetBuffer.Ptr() == nil {
		*targetBuffer = newMat.Clone()
	} else if targetBuffer.Empty() {
		*targetBuffer = newMat.Clone()
	} else {
		newMat.CopyTo(targetBuffer)
	}

	// Swap buffers (atomic)
	if currentIdx == 0 {
		ac.activeIdx = 1
	} else {
		ac.activeIdx = 0
	}

	ac.bufferReady.Store(true)
}

// GetLatestFrame returns a clone of the latest captured frame
// This is non-blocking and returns immediately
func (ac *AsyncCapturer) GetLatestFrame() (*gocv.Mat, error) {
	if !ac.bufferReady.Load() {
		return nil, fmt.Errorf("no frame available yet")
	}

	ac.bufferMu.RLock()
	defer ac.bufferMu.RUnlock()

	// Get active buffer
	var activeBuffer *gocv.Mat
	if ac.activeIdx == 0 {
		activeBuffer = ac.bufferA
	} else {
		activeBuffer = ac.bufferB
	}

	if activeBuffer.Empty() {
		return nil, fmt.Errorf("active buffer is empty")
	}

	// Return a clone so caller can use it safely
	cloned := activeBuffer.Clone()
	return &cloned, nil
}

// ProcessFrame converts a frame to normalized board tensor (reuse from original)
func (ac *AsyncCapturer) ProcessFrame(frame *gocv.Mat) ([]float64, error) {
	if frame.Empty() {
		return nil, fmt.Errorf("empty frame")
	}

	// Convert to grayscale
	gray := gocv.NewMat()
	defer gray.Close()
	gocv.CvtColor(*frame, &gray, gocv.ColorBGRAToGray)

	// Resize to board size
	resized := gocv.NewMat()
	defer resized.Close()
	gocv.Resize(gray, &resized, image.Pt(ac.boardSize, ac.boardSize), 0, 0, gocv.InterpolationLinear)

	// Normalize to [0, 1]
	normalized := gocv.NewMat()
	defer normalized.Close()
	resized.ConvertTo(&normalized, gocv.MatTypeCV32F)
	normalized.DivideFloat(255.0)

	// Convert to float64 slice
	data, err := ac.matToFloat64Slice(&normalized)
	if err != nil {
		return nil, fmt.Errorf("failed to convert mat to slice: %w", err)
	}

	return data, nil
}

// ExtractBoardState captures and processes the current board state
// This is now very fast because it just reads the latest buffer
func (ac *AsyncCapturer) ExtractBoardState() (*BoardState, error) {
	frame, err := ac.GetLatestFrame()
	if err != nil {
		return nil, err
	}
	defer frame.Close()

	// Check for changes
	changed, diffScore, err := ac.detectChange(frame)
	if err != nil {
		return nil, err
	}

	// Process frame to normalized grid
	grid, err := ac.ProcessFrame(frame)
	if err != nil {
		return nil, err
	}

	return &BoardState{
		Grid:      grid,
		Timestamp: time.Now(),
		Changed:   changed,
		DiffScore: diffScore,
	}, nil
}

// detectChange checks if the frame has changed significantly
func (ac *AsyncCapturer) detectChange(frame *gocv.Mat) (bool, float64, error) {
	ac.lastFrameMu.Lock()
	defer ac.lastFrameMu.Unlock()

	if ac.lastFrame == nil {
		// First frame, clone it
		cloned := frame.Clone()
		ac.lastFrame = &cloned
		return true, 100.0, nil
	}

	// Convert both frames to grayscale
	gray1 := gocv.NewMat()
	defer gray1.Close()
	gray2 := gocv.NewMat()
	defer gray2.Close()

	gocv.CvtColor(*ac.lastFrame, &gray1, gocv.ColorBGRAToGray)
	gocv.CvtColor(*frame, &gray2, gocv.ColorBGRAToGray)

	// Compute absolute difference
	diff := gocv.NewMat()
	defer diff.Close()
	gocv.AbsDiff(gray1, gray2, &diff)

	// Calculate mean difference
	mean := diff.Mean()
	meanVal := mean.Val1

	changed := meanVal > ac.diffThreshold

	if changed {
		// Update last frame
		frame.CopyTo(ac.lastFrame)
	}

	return changed, meanVal, nil
}

// GetStatistics returns capture performance statistics
func (ac *AsyncCapturer) GetStatistics() AsyncCaptureStats {
	count := ac.captureCount.Load()
	totalUs := ac.totalCaptureMs.Load()

	var avgMs float64
	if count > 0 {
		avgMs = float64(totalUs) / float64(count) / 1000.0 // convert µs to ms
	}

	return AsyncCaptureStats{
		TotalCaptures: count,
		TotalErrors:   ac.captureErrors.Load(),
		LastCaptureMs: ac.lastCaptureMs.Load(),
		AvgCaptureMs:  avgMs,
		IsRunning:     ac.isRunning.Load(),
		BufferReady:   ac.bufferReady.Load(),
		TargetFPS:     ac.targetFPS,
	}
}

// AsyncCaptureStats holds statistics about async capture performance
type AsyncCaptureStats struct {
	TotalCaptures uint64
	TotalErrors   uint64
	LastCaptureMs int64
	AvgCaptureMs  float64
	IsRunning     bool
	BufferReady   bool
	TargetFPS     int
}

// cleanup releases all resources
func (ac *AsyncCapturer) cleanup() {
	ac.bufferMu.Lock()
	defer ac.bufferMu.Unlock()

	if ac.bufferA != nil && !ac.bufferA.Empty() {
		ac.bufferA.Close()
	}
	if ac.bufferB != nil && !ac.bufferB.Empty() {
		ac.bufferB.Close()
	}

	ac.lastFrameMu.Lock()
	if ac.lastFrame != nil && !ac.lastFrame.Empty() {
		ac.lastFrame.Close()
		ac.lastFrame = nil
	}
	ac.lastFrameMu.Unlock()
}

// Helper methods

func (ac *AsyncCapturer) imageToMat(img image.Image) (*gocv.Mat, error) {
	bounds := img.Bounds()
	width := bounds.Dx()
	height := bounds.Dy()

	mat := gocv.NewMatWithSize(height, width, gocv.MatTypeCV8UC4)

	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			r, g, b, a := img.At(x, y).RGBA()
			mat.SetUCharAt(y, x*4+0, uint8(b>>8))
			mat.SetUCharAt(y, x*4+1, uint8(g>>8))
			mat.SetUCharAt(y, x*4+2, uint8(r>>8))
			mat.SetUCharAt(y, x*4+3, uint8(a>>8))
		}
	}

	return &mat, nil
}

func (ac *AsyncCapturer) matToFloat64Slice(mat *gocv.Mat) ([]float64, error) {
	if mat.Empty() {
		return nil, fmt.Errorf("empty mat")
	}

	rows := mat.Rows()
	cols := mat.Cols()
	data := make([]float64, rows*cols)

	idx := 0
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			data[idx] = float64(mat.GetFloatAt(i, j))
			idx++
		}
	}

	return data, nil
}

// WaitForReady blocks until the first frame is captured
func (ac *AsyncCapturer) WaitForReady(timeout time.Duration) error {
	deadline := time.Now().Add(timeout)

	for time.Now().Before(deadline) {
		if ac.bufferReady.Load() {
			return nil
		}
		time.Sleep(10 * time.Millisecond)
	}

	return fmt.Errorf("timeout waiting for first frame")
}

// IsReady returns true if at least one frame has been captured
func (ac *AsyncCapturer) IsReady() bool {
	return ac.bufferReady.Load()
}
