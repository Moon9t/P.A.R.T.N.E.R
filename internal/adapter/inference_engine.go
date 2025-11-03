package adapter

import (
	"context"
	"fmt"
	"math"
	"sync"
	"time"

	"gorgonia.org/tensor"

	"github.com/thyrook/partner/internal/model"
)

// InferenceEngine provides high-level inference capabilities with adapter integration
// It handles model loading, caching, batch processing, and performance monitoring
type InferenceEngine struct {
	model   *model.ChessCNN
	adapter GameAdapter
	mu      sync.RWMutex

	// Configuration
	config InferenceConfig

	// Performance tracking
	stats InferenceStats

	// Caching for batch operations
	cache     map[string]*CachedPrediction
	cacheMu   sync.RWMutex
	cacheSize int
}

// InferenceConfig holds inference engine configuration
type InferenceConfig struct {
	// Model settings
	ModelPath   string  `json:"model_path"`
	BatchSize   int     `json:"batch_size"`
	MaxCacheAge int     `json:"max_cache_age_sec"` // Cache predictions for repeated states
	EnableCache bool    `json:"enable_cache"`
	Temperature float64 `json:"temperature"` // For softmax temperature scaling

	// Performance
	NumWorkers       int  `json:"num_workers"`       // Parallel inference workers
	TimeoutSec       int  `json:"timeout_sec"`       // Inference timeout
	EnableProfiling  bool `json:"enable_profiling"`  // Detailed performance metrics
	WarmupIterations int  `json:"warmup_iterations"` // Warmup runs before timing
}

// InferenceStats tracks performance metrics
type InferenceStats struct {
	TotalInferences    int64         `json:"total_inferences"`
	SuccessfulInfers   int64         `json:"successful_infers"`
	FailedInfers       int64         `json:"failed_infers"`
	CacheHits          int64         `json:"cache_hits"`
	CacheMisses        int64         `json:"cache_misses"`
	TotalInferenceTime time.Duration `json:"total_inference_time"`
	AvgInferenceTimeMs float64       `json:"avg_inference_time_ms"`
	MinInferenceTimeMs float64       `json:"min_inference_time_ms"`
	MaxInferenceTimeMs float64       `json:"max_inference_time_ms"`
	ThroughputPerSec   float64       `json:"throughput_per_sec"`
	LastUpdateTime     time.Time     `json:"last_update_time"`
}

// CachedPrediction stores a cached inference result
type CachedPrediction struct {
	Result    interface{}
	Timestamp time.Time
	HitCount  int
}

// PredictionResult contains detailed prediction information
type PredictionResult struct {
	Action     interface{}            `json:"action"`
	Confidence float64                `json:"confidence"`
	TopK       []ActionConfidence     `json:"top_k"`
	Latency    time.Duration          `json:"latency"`
	FromCache  bool                   `json:"from_cache"`
	Metadata   map[string]interface{} `json:"metadata"`
	RawOutput  tensor.Tensor          `json:"-"` // Full network output
}

// ActionConfidence represents an action with confidence score
type ActionConfidence struct {
	Action     interface{} `json:"action"`
	Confidence float64     `json:"confidence"`
	Index      int         `json:"index"`
}

// DefaultInferenceConfig returns sensible defaults
func DefaultInferenceConfig() InferenceConfig {
	return InferenceConfig{
		ModelPath:        "data/models/chess_cnn.gob",
		BatchSize:        32,
		MaxCacheAge:      300, // 5 minutes
		EnableCache:      true,
		Temperature:      1.0, // No scaling
		NumWorkers:       4,
		TimeoutSec:       30,
		EnableProfiling:  true,
		WarmupIterations: 5,
	}
}

// NewInferenceEngine creates a new inference engine
func NewInferenceEngine(adapter GameAdapter, config InferenceConfig) (*InferenceEngine, error) {
	if adapter == nil {
		return nil, fmt.Errorf("adapter cannot be nil")
	}

	engine := &InferenceEngine{
		adapter:   adapter,
		config:    config,
		cache:     make(map[string]*CachedPrediction),
		cacheSize: 1000, // Max cache entries
		stats: InferenceStats{
			MinInferenceTimeMs: 999999.0,
			LastUpdateTime:     time.Now(),
		},
	}

	return engine, nil
}

// LoadModel loads the neural network model from a checkpoint file
func (ie *InferenceEngine) LoadModel(modelPath string) error {
	ie.mu.Lock()
	defer ie.mu.Unlock()

	// Create model for inference (batch size 1)
	loadedModel, err := model.NewChessCNNForInference(modelPath)
	if err != nil {
		return fmt.Errorf("failed to load model: %w", err)
	}

	ie.model = loadedModel
	ie.config.ModelPath = modelPath

	// Run warmup if configured
	if ie.config.WarmupIterations > 0 {
		if err := ie.warmup(); err != nil {
			return fmt.Errorf("warmup failed: %w", err)
		}
	}

	return nil
}

// Predict performs inference on a single state
func (ie *InferenceEngine) Predict(ctx context.Context, state interface{}) (*PredictionResult, error) {
	startTime := time.Now()

	// Check if model is loaded
	ie.mu.RLock()
	if ie.model == nil {
		ie.mu.RUnlock()
		return nil, fmt.Errorf("model not loaded")
	}
	ie.mu.RUnlock()

	// Generate cache key if caching is enabled
	var cacheKey string
	if ie.config.EnableCache {
		cacheKey = ie.generateCacheKey(state)
		if cached := ie.getCached(cacheKey); cached != nil {
			// Return cached result
			ie.updateStats(time.Since(startTime), true, true)

			// Cast cached result to action
			cachedAction, ok := cached.(map[string]interface{})
			if !ok {
				cachedAction = map[string]interface{}{"cached": cached}
			}

			return &PredictionResult{
				Action:     cachedAction,
				Confidence: 1.0, // Cached results are deterministic
				Latency:    time.Since(startTime),
				FromCache:  true,
			}, nil
		}
	}

	// Encode state using adapter
	stateTensor, err := ie.adapter.EncodeState(state)
	if err != nil {
		ie.updateStats(time.Since(startTime), false, false)
		return nil, fmt.Errorf("failed to encode state: %w", err)
	}

	// Run inference with timeout
	resultChan := make(chan *PredictionResult, 1)
	errorChan := make(chan error, 1)

	go func() {
		result, err := ie.runInference(stateTensor, startTime)
		if err != nil {
			errorChan <- err
			return
		}
		resultChan <- result
	}()

	// Wait for result or timeout
	timeoutDuration := time.Duration(ie.config.TimeoutSec) * time.Second
	select {
	case <-ctx.Done():
		return nil, fmt.Errorf("inference cancelled: %w", ctx.Err())
	case <-time.After(timeoutDuration):
		ie.updateStats(time.Since(startTime), false, false)
		return nil, fmt.Errorf("inference timeout after %v", timeoutDuration)
	case err := <-errorChan:
		ie.updateStats(time.Since(startTime), false, false)
		return nil, err
	case result := <-resultChan:
		ie.updateStats(time.Since(startTime), true, false)

		// Cache result if enabled
		if ie.config.EnableCache && cacheKey != "" {
			ie.cacheResult(cacheKey, result.Action)
		}

		return result, nil
	}
}

// PredictBatch performs inference on multiple states efficiently
func (ie *InferenceEngine) PredictBatch(ctx context.Context, states []interface{}) ([]*PredictionResult, error) {
	if len(states) == 0 {
		return nil, fmt.Errorf("no states provided")
	}

	results := make([]*PredictionResult, len(states))
	errors := make([]error, len(states))

	// Use worker pool for parallel processing
	numWorkers := ie.config.NumWorkers
	if numWorkers <= 0 {
		numWorkers = 1
	}
	if numWorkers > len(states) {
		numWorkers = len(states)
	}

	// Create job channel
	jobs := make(chan int, len(states))
	var wg sync.WaitGroup

	// Start workers
	for w := 0; w < numWorkers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for idx := range jobs {
				result, err := ie.Predict(ctx, states[idx])
				if err != nil {
					errors[idx] = err
				} else {
					results[idx] = result
				}
			}
		}()
	}

	// Send jobs
	for i := range states {
		jobs <- i
	}
	close(jobs)

	// Wait for completion
	wg.Wait()

	// Check for errors
	var firstError error
	for _, err := range errors {
		if err != nil && firstError == nil {
			firstError = err
		}
	}

	return results, firstError
}

// runInference performs the actual model inference
func (ie *InferenceEngine) runInference(stateTensor tensor.Tensor, startTime time.Time) (*PredictionResult, error) {
	ie.mu.RLock()
	defer ie.mu.RUnlock()

	// Convert tensor to [12][8][8]float32 format expected by ChessCNN
	boardTensor, err := ie.tensorToBoardArray(stateTensor)
	if err != nil {
		return nil, fmt.Errorf("failed to convert tensor: %w", err)
	}

	// Run model prediction
	predictions, err := ie.model.Predict(boardTensor, 10) // Get top 10 moves
	if err != nil {
		return nil, fmt.Errorf("model prediction failed: %w", err)
	}

	if len(predictions) == 0 {
		return nil, fmt.Errorf("no predictions returned from model")
	}

	// Apply temperature scaling to probabilities if configured
	if ie.config.Temperature != 1.0 {
		predictions = ie.applyTemperatureToMoves(predictions, ie.config.Temperature)
	}

	// Get top move
	topMove := predictions[0]

	// Convert to output tensor format for adapter
	outputProbs := make([]float64, 4096)
	for _, pred := range predictions {
		outputProbs[pred.MoveIndex] = pred.Probability
	}

	outputTensor := tensor.New(
		tensor.WithShape(4096),
		tensor.WithBacking(outputProbs),
	)

	// Decode action using adapter
	action, err := ie.adapter.DecodeAction(outputTensor)
	if err != nil {
		return nil, fmt.Errorf("failed to decode action: %w", err)
	}

	// Build top-K actions
	topK := make([]ActionConfidence, 0, len(predictions))
	for i, pred := range predictions {
		topK = append(topK, ActionConfidence{
			Action:     fmt.Sprintf("%s%s", squareToAlg(pred.FromSquare), squareToAlg(pred.ToSquare)),
			Confidence: pred.Probability,
			Index:      pred.MoveIndex,
		})
		if i >= 4 { // Limit to top 5
			break
		}
	}

	result := &PredictionResult{
		Action:     action,
		Confidence: topMove.Probability,
		TopK:       topK,
		Latency:    time.Since(startTime),
		FromCache:  false,
		RawOutput:  outputTensor,
		Metadata: map[string]interface{}{
			"from_square": topMove.FromSquare,
			"to_square":   topMove.ToSquare,
			"move_index":  topMove.MoveIndex,
		},
	}

	return result, nil
}

// warmup runs dummy inferences to warm up the model
func (ie *InferenceEngine) warmup() error {
	// Create a dummy chess board (starting position)
	var dummyBoard [12][8][8]float32

	// Set up a starting position with a few pieces
	dummyBoard[0][1][0] = 1.0  // White pawn on a2
	dummyBoard[0][1][7] = 1.0  // White pawn on h2
	dummyBoard[5][0][4] = 1.0  // White king on e1
	dummyBoard[6][6][0] = 1.0  // Black pawn on a7
	dummyBoard[11][7][4] = 1.0 // Black king on e8

	// Run warmup iterations
	for i := 0; i < ie.config.WarmupIterations; i++ {
		_, err := ie.model.Predict(dummyBoard, 3)
		if err != nil {
			return fmt.Errorf("warmup iteration %d failed: %w", i+1, err)
		}
	}

	return nil
}

// tensorToBoardArray converts a tensor to [12][8][8]float32 board representation
func (ie *InferenceEngine) tensorToBoardArray(t tensor.Tensor) ([12][8][8]float32, error) {
	var board [12][8][8]float32

	shape := t.Shape()
	data := t.Data().([]float64)

	// Handle different input shapes
	var channels, height, width int

	if len(shape) == 3 {
		// Shape: [12, 8, 8]
		channels, height, width = shape[0], shape[1], shape[2]
	} else if len(shape) == 4 {
		// Shape: [batch, 12, 8, 8] - take first item
		channels, height, width = shape[1], shape[2], shape[3]
	} else {
		return board, fmt.Errorf("unsupported tensor shape: %v", shape)
	}

	if channels != 12 || height != 8 || width != 8 {
		return board, fmt.Errorf("expected [12, 8, 8] dimensions, got [%d, %d, %d]", channels, height, width)
	}

	// Copy data
	idx := 0
	for c := 0; c < 12; c++ {
		for h := 0; h < 8; h++ {
			for w := 0; w < 8; w++ {
				board[c][h][w] = float32(data[idx])
				idx++
			}
		}
	}

	return board, nil
}

// applyTemperatureToMoves applies temperature scaling to move probabilities
func (ie *InferenceEngine) applyTemperatureToMoves(moves []model.MovePrediction, temperature float64) []model.MovePrediction {
	if temperature == 1.0 {
		return moves
	}

	// Apply temperature scaling: prob = prob^(1/T) / sum(prob^(1/T))
	scaled := make([]model.MovePrediction, len(moves))
	sum := 0.0

	for i, move := range moves {
		scaledProb := math.Pow(move.Probability, 1.0/temperature)
		scaled[i] = move
		scaled[i].Probability = scaledProb
		sum += scaledProb
	}

	// Normalize
	if sum > 0 {
		for i := range scaled {
			scaled[i].Probability /= sum
		}
	}

	return scaled
}

// squareToAlg converts square index (0-63) to algebraic notation
func squareToAlg(square int) string {
	if square < 0 || square >= 64 {
		return "??"
	}
	file := square % 8
	rank := square / 8
	return string(rune('a'+file)) + string(rune('1'+rank))
}

// generateCacheKey generates a cache key from state
func (ie *InferenceEngine) generateCacheKey(state interface{}) string {
	// Simple string representation for now
	// In production, use a proper hash function
	return fmt.Sprintf("%v", state)
}

// getCached retrieves a cached prediction
func (ie *InferenceEngine) getCached(key string) interface{} {
	ie.cacheMu.RLock()
	defer ie.cacheMu.RUnlock()

	cached, exists := ie.cache[key]
	if !exists {
		return nil
	}

	// Check age
	age := time.Since(cached.Timestamp).Seconds()
	if age > float64(ie.config.MaxCacheAge) {
		return nil // Expired
	}

	cached.HitCount++
	return cached.Result
}

// cacheResult stores a prediction in cache
func (ie *InferenceEngine) cacheResult(key string, result interface{}) {
	ie.cacheMu.Lock()
	defer ie.cacheMu.Unlock()

	// Evict old entries if cache is full
	if len(ie.cache) >= ie.cacheSize {
		ie.evictOldest()
	}

	ie.cache[key] = &CachedPrediction{
		Result:    result,
		Timestamp: time.Now(),
		HitCount:  0,
	}
}

// evictOldest removes the oldest cache entry
func (ie *InferenceEngine) evictOldest() {
	var oldestKey string
	var oldestTime time.Time

	for key, cached := range ie.cache {
		if oldestKey == "" || cached.Timestamp.Before(oldestTime) {
			oldestKey = key
			oldestTime = cached.Timestamp
		}
	}

	if oldestKey != "" {
		delete(ie.cache, oldestKey)
	}
}

// updateStats updates performance statistics
func (ie *InferenceEngine) updateStats(latency time.Duration, success bool, fromCache bool) {
	ie.mu.Lock()
	defer ie.mu.Unlock()

	ie.stats.TotalInferences++

	if success {
		ie.stats.SuccessfulInfers++
	} else {
		ie.stats.FailedInfers++
	}

	if fromCache {
		ie.stats.CacheHits++
	} else {
		ie.stats.CacheMisses++
	}

	latencyMs := float64(latency.Microseconds()) / 1000.0
	ie.stats.TotalInferenceTime += latency

	// Update average
	if ie.stats.SuccessfulInfers > 0 {
		ie.stats.AvgInferenceTimeMs = float64(ie.stats.TotalInferenceTime.Milliseconds()) / float64(ie.stats.SuccessfulInfers)
	}

	// Update min/max
	if latencyMs < ie.stats.MinInferenceTimeMs {
		ie.stats.MinInferenceTimeMs = latencyMs
	}
	if latencyMs > ie.stats.MaxInferenceTimeMs {
		ie.stats.MaxInferenceTimeMs = latencyMs
	}

	// Calculate throughput
	elapsed := time.Since(ie.stats.LastUpdateTime).Seconds()
	if elapsed > 0 {
		ie.stats.ThroughputPerSec = float64(ie.stats.TotalInferences) / elapsed
	}
}

// GetStats returns current performance statistics
func (ie *InferenceEngine) GetStats() InferenceStats {
	ie.mu.RLock()
	defer ie.mu.RUnlock()
	return ie.stats
}

// ResetStats resets performance statistics
func (ie *InferenceEngine) ResetStats() {
	ie.mu.Lock()
	defer ie.mu.Unlock()

	ie.stats = InferenceStats{
		MinInferenceTimeMs: 999999.0,
		LastUpdateTime:     time.Now(),
	}
}

// ClearCache clears the prediction cache
func (ie *InferenceEngine) ClearCache() {
	ie.cacheMu.Lock()
	defer ie.cacheMu.Unlock()

	ie.cache = make(map[string]*CachedPrediction)
}

// GetCacheStats returns cache statistics
func (ie *InferenceEngine) GetCacheStats() map[string]interface{} {
	ie.cacheMu.RLock()
	defer ie.cacheMu.RUnlock()

	totalHits := int64(0)
	for _, cached := range ie.cache {
		totalHits += int64(cached.HitCount)
	}

	hitRate := 0.0
	total := ie.stats.CacheHits + ie.stats.CacheMisses
	if total > 0 {
		hitRate = float64(ie.stats.CacheHits) / float64(total)
	}

	return map[string]interface{}{
		"size":       len(ie.cache),
		"max_size":   ie.cacheSize,
		"total_hits": totalHits,
		"hit_rate":   hitRate,
	}
}

// Close cleans up resources
func (ie *InferenceEngine) Close() error {
	ie.mu.Lock()
	defer ie.mu.Unlock()

	if ie.model != nil {
		ie.model.Close()
		ie.model = nil
	}

	ie.ClearCache()
	return nil
}
