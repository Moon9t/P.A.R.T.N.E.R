package adapter

// This file contains usage examples for the InferenceEngine.
// These are documentation examples and are not compiled into the binary.

/*

=== INFERENCE ENGINE USAGE EXAMPLES ===

The InferenceEngine provides high-level inference capabilities with adapter integration,
model loading, caching, batch processing, and performance monitoring.

--- Example 1: Basic Chess Inference ---

	package main

	import (
		"context"
		"fmt"
		"log"

		"github.com/thyrook/partner/internal/adapter"
	)

	func main() {
		// Create chess adapter
		chessAdapter := adapter.NewChessAdapter()

		// Create inference engine with default config
		config := adapter.DefaultInferenceConfig()
		config.ModelPath = "data/models/chess_cnn.gob"
		config.EnableCache = true
		config.Temperature = 1.0

		engine, err := adapter.NewInferenceEngine(chessAdapter, config)
		if err != nil {
			log.Fatal(err)
		}
		defer engine.Close()

		// Load trained model
		if err := engine.LoadModel(config.ModelPath); err != nil {
			log.Fatal(err)
		}

		// Create a chess position (starting position FEN)
		fen := "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

		// Run inference
		ctx := context.Background()
		result, err := engine.Predict(ctx, fen)
		if err != nil {
			log.Fatal(err)
		}

		// Print results
		fmt.Printf("Best move: %v\n", result.Action)
		fmt.Printf("Confidence: %.2f%%\n", result.Confidence*100)
		fmt.Printf("Latency: %v\n", result.Latency)
		fmt.Printf("From cache: %v\n", result.FromCache)

		// Print top 5 moves
		fmt.Println("\nTop moves:")
		for _, move := range result.TopK {
			fmt.Printf("  %v (%.2f%%)\n", move.Action, move.Confidence*100)
		}

		// Get statistics
		stats := engine.GetStats()
		fmt.Printf("\nInference stats:\n")
		fmt.Printf("  Total inferences: %d\n", stats.TotalInferences)
		fmt.Printf("  Cache hit rate: %.2f%%\n",
			float64(stats.CacheHits)/float64(stats.TotalInferences)*100)
		fmt.Printf("  Avg latency: %.2fms\n", stats.AvgInferenceTimeMs)
	}


--- Example 2: Batch Inference with Statistics ---

	func batchInferenceExample() {
		// Setup
		adapter := adapter.NewChessAdapter()
		config := adapter.DefaultInferenceConfig()
		config.NumWorkers = 8 // Parallel workers

		engine, _ := adapter.NewInferenceEngine(adapter, config)
		engine.LoadModel("data/models/chess_cnn.gob")

		// Prepare batch of positions
		positions := []interface{}{
			"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
			"rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
			"rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2",
		}

		// Run batch inference
		ctx := context.Background()
		results, err := engine.PredictBatch(ctx, positions)
		if err != nil {
			log.Printf("Batch error: %v", err)
		}

		// Analyze results
		totalConfidence := 0.0
		for i, result := range results {
			if result != nil {
				fmt.Printf("Position %d: %v (%.2f%%)\n",
					i+1, result.Action, result.Confidence*100)
				totalConfidence += result.Confidence
			}
		}

		avgConfidence := totalConfidence / float64(len(results))
		fmt.Printf("Average confidence: %.2f%%\n", avgConfidence*100)

		// Performance stats
		stats := engine.GetStats()
		fmt.Printf("Throughput: %.1f inferences/sec\n", stats.ThroughputPerSec)
	}


--- Example 3: Racing Game Inference ---

	func racingInferenceExample() {
		// Create racing adapter
		racingAdapter := adapter.NewRacingAdapter()

		config := adapter.DefaultInferenceConfig()
		config.ModelPath = "data/models/racing_model.gob"
		config.Temperature = 0.8 // Slightly more exploratory

		engine, _ := adapter.NewInferenceEngine(racingAdapter, config)
		engine.LoadModel(config.ModelPath)

		// Create racing game state
		state := adapter.RacingState{
			Speed:    180.0, // km/h
			Position: adapter.Position{X: 100.0, Y: 250.0, Heading: 1.57},
			TrackSensors: []float64{
				15.0, 18.0, 20.0, 25.0, 22.0, 19.0, 16.0, 14.0,
			},
			OnTrack: true,
		}

		// Run inference
		ctx := context.Background()
		result, err := engine.Predict(ctx, state)
		if err != nil {
			log.Fatal(err)
		}

		// Extract controls
		controls := result.Action.(map[string]interface{})
		fmt.Printf("Controls:\n")
		fmt.Printf("  Steering: %.2f\n", controls["steering"])
		fmt.Printf("  Throttle: %.2f\n", controls["throttle"])
		fmt.Printf("  Brake: %.2f\n", controls["brake"])
		fmt.Printf("Confidence: %.2f%%\n", result.Confidence*100)
	}


--- Example 4: Advanced Configuration with Monitoring ---

	func advancedInferenceExample() {
		adapter := adapter.NewChessAdapter()

		// Custom configuration
		config := adapter.InferenceConfig{
			ModelPath:        "data/models/chess_cnn.gob",
			BatchSize:        64,
			MaxCacheAge:      600, // 10 minutes
			EnableCache:      true,
			Temperature:      1.2, // More exploratory
			NumWorkers:       16,
			TimeoutSec:       10,
			EnableProfiling:  true,
			WarmupIterations: 10,
		}

		engine, _ := adapter.NewInferenceEngine(adapter, config)
		engine.LoadModel(config.ModelPath)

		// Run many inferences
		ctx := context.Background()
		for i := 0; i < 1000; i++ {
			fen := generateRandomPosition() // Your function
			result, err := engine.Predict(ctx, fen)
			if err != nil {
				continue
			}

			// Process result
			_ = result
		}

		// Detailed statistics
		stats := engine.GetStats()
		fmt.Printf("Performance Report:\n")
		fmt.Printf("  Total inferences: %d\n", stats.TotalInferences)
		fmt.Printf("  Success rate: %.2f%%\n",
			float64(stats.SuccessfulInfers)/float64(stats.TotalInferences)*100)
		fmt.Printf("  Cache hit rate: %.2f%%\n",
			float64(stats.CacheHits)/float64(stats.TotalInferences)*100)
		fmt.Printf("  Avg latency: %.2fms\n", stats.AvgInferenceTimeMs)
		fmt.Printf("  Min latency: %.2fms\n", stats.MinInferenceTimeMs)
		fmt.Printf("  Max latency: %.2fms\n", stats.MaxInferenceTimeMs)
		fmt.Printf("  Throughput: %.1f/sec\n", stats.ThroughputPerSec)

		// Cache statistics
		cacheStats := engine.GetCacheStats()
		fmt.Printf("\nCache Statistics:\n")
		fmt.Printf("  Size: %d/%d\n", cacheStats["size"], cacheStats["max_size"])
		fmt.Printf("  Hit rate: %.2f%%\n", cacheStats["hit_rate"].(float64)*100)
		fmt.Printf("  Total hits: %d\n", cacheStats["total_hits"])

		// Reset stats for new benchmark
		engine.ResetStats()

		// Clear cache if needed
		engine.ClearCache()
	}


--- Example 5: Custom Adapter Integration ---

	// Create your own game adapter
	type MyGameAdapter struct {
		*adapter.BaseAdapter
	}

	func NewMyGameAdapter() *MyGameAdapter {
		return &MyGameAdapter{
			BaseAdapter: adapter.NewBaseAdapter("mygame", []int{10, 10}, []int{4}),
		}
	}

	func (m *MyGameAdapter) EncodeState(frame interface{}) (tensor.Tensor, error) {
		// Your encoding logic
		return tensor.New(tensor.WithShape(10, 10), tensor.WithBacking(data)), nil
	}

	func (m *MyGameAdapter) DecodeAction(pred tensor.Tensor) (interface{}, error) {
		// Your decoding logic
		return myAction, nil
	}

	// ... implement other GameAdapter methods

	func customAdapterExample() {
		// Use your custom adapter
		myAdapter := NewMyGameAdapter()
		config := adapter.DefaultInferenceConfig()

		engine, _ := adapter.NewInferenceEngine(myAdapter, config)
		engine.LoadModel("data/models/my_game.gob")

		// Now use it just like any other adapter
		result, _ := engine.Predict(context.Background(), myGameState)
		fmt.Printf("Action: %v\n", result.Action)
	}


--- Example 6: Error Handling and Timeouts ---

	func robustInferenceExample() {
		adapter := adapter.NewChessAdapter()
		config := adapter.DefaultInferenceConfig()
		config.TimeoutSec = 5 // 5 second timeout

		engine, err := adapter.NewInferenceEngine(adapter, config)
		if err != nil {
			log.Fatal("Failed to create engine:", err)
		}
		defer engine.Close()

		// Load with error handling
		if err := engine.LoadModel("data/models/chess_cnn.gob"); err != nil {
			log.Fatal("Failed to load model:", err)
		}

		// Inference with timeout
		ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
		defer cancel()

		result, err := engine.Predict(ctx, someFEN)
		if err != nil {
			if err == context.DeadlineExceeded {
				log.Println("Inference timed out")
			} else {
				log.Println("Inference failed:", err)
			}
			return
		}

		// Check confidence threshold
		if result.Confidence < 0.5 {
			log.Printf("Low confidence: %.2f%%, using fallback\n", result.Confidence*100)
			// Use fallback strategy
		} else {
			// Use prediction
			fmt.Printf("High confidence move: %v\n", result.Action)
		}
	}


=== CONFIGURATION OPTIONS ===

InferenceConfig fields:
  - ModelPath: Path to trained model checkpoint
  - BatchSize: Batch size for batch inference
  - MaxCacheAge: Cache entry TTL in seconds
  - EnableCache: Enable prediction caching
  - Temperature: Softmax temperature (1.0=no scaling, <1=sharper, >1=softer)
  - NumWorkers: Parallel workers for batch inference
  - TimeoutSec: Per-inference timeout
  - EnableProfiling: Detailed performance metrics
  - WarmupIterations: Number of warmup runs

=== PERFORMANCE TIPS ===

1. Enable caching for repeated positions (chess openings, common states)
2. Use batch inference for multiple predictions (much faster than sequential)
3. Set appropriate timeout values
4. Use temperature scaling for exploration vs exploitation
5. Monitor cache hit rate - if low, increase cache size or TTL
6. Use more workers for I/O-bound batch operations
7. Profile with EnableProfiling=true to find bottlenecks

=== COMMON PATTERNS ===

Chess Engine:
  - Temperature: 1.0 (deterministic)
  - Cache: Enabled (openings repeat)
  - BatchSize: 32-64

Racing AI:
  - Temperature: 0.8-1.2 (exploration)
  - Cache: Disabled (states don't repeat)
  - NumWorkers: High (many parallel predictions)

Analysis Tool:
  - BatchSize: Large (analyzing games)
  - EnableProfiling: true
  - NumWorkers: Max available

*/
