package vision

import (
	"fmt"
	"sync"
	"time"

	"gocv.io/x/gocv"
)

// Pipeline manages the complete vision processing pipeline
type Pipeline struct {
	config       *Config
	source       FrameSource
	capturer     *Capturer
	detector     *BoardDetector
	lastBoard    *[12][8][8]float32
	tensorChan   chan<- BoardStateTensor
	stopChan     chan struct{}
	wg           sync.WaitGroup
	mu           sync.Mutex
	running      bool
	stats        PipelineStats
}

// PipelineStats tracks pipeline performance
type PipelineStats struct {
	FramesProcessed  int64
	ChangesDetected  int64
	LastProcessTime  time.Duration
	AverageFrameTime time.Duration
	Errors           int64
}

// NewPipeline creates a new vision pipeline
func NewPipeline(config *Config, tensorChan chan<- BoardStateTensor) (*Pipeline, error) {
	if err := config.Validate(); err != nil {
		return nil, fmt.Errorf("invalid config: %w", err)
	}

	// Create capturer
	capturer := NewCapturer(
		config.CaptureRegion.X,
		config.CaptureRegion.Y,
		config.CaptureRegion.Width,
		config.CaptureRegion.Height,
		config.BoardSize,
		config.DiffThreshold,
	)

	// Create board detector
	detector := NewBoardDetector(config.SquareSize, config.UseGrayscale)

	// Set custom color thresholds if provided
	if config.ColorThresholds != nil {
		detector.colorThresholds = *config.ColorThresholds
	}

	return &Pipeline{
		config:     config,
		capturer:   capturer,
		detector:   detector,
		tensorChan: tensorChan,
		stopChan:   make(chan struct{}),
	}, nil
}

// NewPipelineWithVideo creates a pipeline using a video file as source
func NewPipelineWithVideo(config *Config, videoPath string, tensorChan chan<- BoardStateTensor) (*Pipeline, error) {
	pipeline, err := NewPipeline(config, tensorChan)
	if err != nil {
		return nil, err
	}

	// Replace with video source
	videoSource, err := NewVideoSource(videoPath)
	if err != nil {
		return nil, fmt.Errorf("failed to create video source: %w", err)
	}

	pipeline.source = videoSource
	return pipeline, nil
}

// Start begins the vision processing pipeline
func (p *Pipeline) Start() error {
	p.mu.Lock()
	if p.running {
		p.mu.Unlock()
		return fmt.Errorf("pipeline already running")
	}
	p.running = true
	p.mu.Unlock()

	// Determine frame source
	if p.source == nil {
		// Use live capture
		p.source = NewLiveSource(p.capturer)
	}

	// Start processing loop
	p.wg.Add(1)
	go p.processLoop()

	return nil
}

// Stop stops the pipeline
func (p *Pipeline) Stop() {
	p.mu.Lock()
	if !p.running {
		p.mu.Unlock()
		return
	}
	p.mu.Unlock()

	close(p.stopChan)
	p.wg.Wait()

	p.mu.Lock()
	p.running = false
	p.mu.Unlock()
}

// IsRunning returns whether the pipeline is running
func (p *Pipeline) IsRunning() bool {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.running
}

// GetStats returns current pipeline statistics
func (p *Pipeline) GetStats() PipelineStats {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.stats
}

// processLoop is the main processing loop
func (p *Pipeline) processLoop() {
	defer p.wg.Done()

	frameDuration := time.Second / time.Duration(p.config.FPS)
	ticker := time.NewTicker(frameDuration)
	defer ticker.Stop()

	for {
		select {
		case <-p.stopChan:
			return
		case <-ticker.C:
			if err := p.processSingleFrame(); err != nil {
				p.mu.Lock()
				p.stats.Errors++
				p.mu.Unlock()
				// Log error but continue
				fmt.Printf("Frame processing error: %v\n", err)
			}
		}
	}
}

// processSingleFrame processes one frame
func (p *Pipeline) processSingleFrame() error {
	startTime := time.Now()

	// Read frame from source
	frame, err := p.source.ReadFrame()
	if err != nil {
		return fmt.Errorf("failed to read frame: %w", err)
	}
	defer frame.Close()

	// Check if frame changed significantly
	changed, _, err := p.capturer.DetectChange(frame)
	if err != nil {
		return fmt.Errorf("failed to detect change: %w", err)
	}

	// Update stats
	p.mu.Lock()
	p.stats.FramesProcessed++
	processingTime := time.Since(startTime)
	p.stats.LastProcessTime = processingTime

	// Update average (simple moving average)
	if p.stats.AverageFrameTime == 0 {
		p.stats.AverageFrameTime = processingTime
	} else {
		p.stats.AverageFrameTime = (p.stats.AverageFrameTime + processingTime) / 2
	}
	p.mu.Unlock()

	// Only process if significant change detected
	if !changed {
		return nil
	}

	// Detect board state
	boardTensor, err := p.detector.DetectBoard(frame)
	if err != nil {
		return fmt.Errorf("failed to detect board: %w", err)
	}

	// Validate tensor
	if err := ValidateBoardTensor(boardTensor); err != nil {
		// Log validation error but don't fail
		fmt.Printf("Board validation warning: %v\n", err)
	}

	// Detect changes from last board state
	var changes []Position
	if p.lastBoard != nil {
		changes = p.detector.DetectBoardDifference(*p.lastBoard, boardTensor)
	}

	// Update last board
	p.lastBoard = &boardTensor

	// Send tensor to channel
	tensorMsg := BoardStateTensor{
		Tensor:    boardTensor,
		Timestamp: time.Now().Unix(),
		Changes:   changes,
	}

	select {
	case p.tensorChan <- tensorMsg:
		p.mu.Lock()
		p.stats.ChangesDetected++
		p.mu.Unlock()
	case <-p.stopChan:
		return nil
	}

	return nil
}

// ProcessSingleImage processes a single image file for testing
func (p *Pipeline) ProcessSingleImage(imagePath string) (*BoardStateTensor, error) {
	// Load image
	img := gocv.IMRead(imagePath, gocv.IMReadColor)
	if img.Empty() {
		return nil, fmt.Errorf("failed to load image: %s", imagePath)
	}
	defer img.Close()

	// Detect board
	boardTensor, err := p.detector.DetectBoard(&img)
	if err != nil {
		return nil, fmt.Errorf("failed to detect board: %w", err)
	}

	// Validate
	if err := ValidateBoardTensor(boardTensor); err != nil {
		return nil, fmt.Errorf("invalid board tensor: %w", err)
	}

	return &BoardStateTensor{
		Tensor:    boardTensor,
		Timestamp: time.Now().Unix(),
	}, nil
}

// Close releases all resources
func (p *Pipeline) Close() error {
	p.Stop()

	if p.source != nil {
		p.source.Close()
	}

	p.capturer.Close()

	return nil
}

// String returns pipeline status
func (p *Pipeline) String() string {
	stats := p.GetStats()
	return fmt.Sprintf(
		"Vision Pipeline:\n"+
			"  Running: %v\n"+
			"  Frames Processed: %d\n"+
			"  Changes Detected: %d\n"+
			"  Last Process Time: %v\n"+
			"  Avg Frame Time: %v\n"+
			"  Errors: %d\n"+
			"  FPS Target: %d\n",
		p.IsRunning(),
		stats.FramesProcessed,
		stats.ChangesDetected,
		stats.LastProcessTime,
		stats.AverageFrameTime,
		stats.Errors,
		p.config.FPS,
	)
}
