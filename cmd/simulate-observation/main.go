package main

import (
	"flag"
	"fmt"
	"math/rand"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/thyrook/partner/internal/storage"
)

func main() {
	// Parse command line flags
	dbPath := flag.String("db", "data/observations.db", "Path to observations database")
	maxSize := flag.Int("max-size", 10000, "Maximum number of samples to store")
	numSamples := flag.Int("samples", 100, "Number of samples to generate")
	batchSize := flag.Int("batch", 32, "Batch size for testing retrieval")
	clearFirst := flag.Bool("clear", false, "Clear existing data before starting")
	testOnly := flag.Bool("test", false, "Only test retrieval, don't generate new samples")
	flag.Parse()

	fmt.Println("═══════════════════════════════════════════════════════════")
	fmt.Println("  P.A.R.T.N.E.R - Observation Storage Simulation")
	fmt.Println("═══════════════════════════════════════════════════════════")
	fmt.Println()

	// Ensure data directory exists
	if err := os.MkdirAll("data", 0755); err != nil {
		fmt.Fprintf(os.Stderr, "❌ Failed to create data directory: %v\n", err)
		os.Exit(1)
	}

	// Initialize observation store
	fmt.Printf("Opening observation store: %s\n", *dbPath)
	store, err := storage.NewObservationStore(*dbPath, *maxSize)
	if err != nil {
		fmt.Fprintf(os.Stderr, "❌ Failed to open store: %v\n", err)
		os.Exit(1)
	}
	defer store.Close()

	fmt.Println("✓ Observation store ready")
	fmt.Println()

	// Display initial stats
	displayStats(store)

	// Clear if requested
	if *clearFirst {
		fmt.Println("Clearing existing data...")
		if err := store.Clear(); err != nil {
			fmt.Fprintf(os.Stderr, "❌ Failed to clear store: %v\n", err)
			os.Exit(1)
		}
		fmt.Println("✓ Data cleared")
		fmt.Println()
		displayStats(store)
	}

	// Test mode: only retrieve and display samples
	if *testOnly {
		testRetrieval(store, *batchSize)
		return
	}

	// Setup signal handling for graceful shutdown
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, os.Interrupt, syscall.SIGTERM)

	// Simulate observation collection
	fmt.Printf("Simulating observation of %d chess positions...\n", *numSamples)
	fmt.Println("(Press Ctrl+C to stop early)")
	fmt.Println()

	collected := 0
	startTime := time.Now()

	// Collection loop
	ticker := time.NewTicker(100 * time.Millisecond)
	defer ticker.Stop()

	for collected < *numSamples {
		select {
		case <-sigChan:
			fmt.Println("\n\n⚠️  Interrupted by user")
			goto done
		case <-ticker.C:
			// Simulate observing a chess position
			sample := generateRandomChessSample()

			// Store the observation
			if err := store.StoreSample(sample.state, sample.move); err != nil {
				fmt.Fprintf(os.Stderr, "❌ Failed to store sample: %v\n", err)
				continue
			}

			collected++

			// Show progress every 10 samples
			if collected%10 == 0 {
				elapsed := time.Since(startTime)
				rate := float64(collected) / elapsed.Seconds()
				fmt.Printf("\rProgress: %d/%d samples (%.1f samples/sec)",
					collected, *numSamples, rate)
			}
		}
	}

done:
	duration := time.Since(startTime)
	fmt.Printf("\n\n✓ Collected %d samples in %v\n", collected, duration.Round(time.Millisecond))
	fmt.Printf("  Average rate: %.1f samples/second\n", float64(collected)/duration.Seconds())
	fmt.Println()

	// Display final stats
	displayStats(store)

	// Test batch retrieval
	fmt.Println()
	testRetrieval(store, *batchSize)

	fmt.Println()
	fmt.Println("═══════════════════════════════════════════════════════════")
	fmt.Println("  Simulation Complete")
	fmt.Println("═══════════════════════════════════════════════════════════")
}

// chessSample represents a simulated chess observation
type chessSample struct {
	state []float64
	move  int
}

// generateRandomChessSample generates a random chess position and move
func generateRandomChessSample() chessSample {
	// Create 8x8 board state (64 values)
	state := make([]float64, 64)

	// Simulate piece positions with values:
	// 0.0 = empty
	// 0.1-0.6 = white pieces (pawn, knight, bishop, rook, queen, king)
	// 0.7-1.0 = black pieces

	// Randomly place some pieces
	numPieces := 16 + rand.Intn(16) // 16-32 pieces on board
	for i := 0; i < numPieces; i++ {
		square := rand.Intn(64)
		pieceType := rand.Float64()

		// Make it more chess-like (pawns more common, kings rare)
		if rand.Float64() < 0.4 {
			// Pawn
			if rand.Float64() < 0.5 {
				state[square] = 0.1 // White pawn
			} else {
				state[square] = 0.7 // Black pawn
			}
		} else if rand.Float64() < 0.2 {
			// Minor piece
			if rand.Float64() < 0.5 {
				state[square] = 0.2 + rand.Float64()*0.1 // White minor
			} else {
				state[square] = 0.8 + rand.Float64()*0.1 // Black minor
			}
		} else {
			// Other pieces
			state[square] = pieceType
		}
	}

	// Generate random move (from square, to square encoded as integer)
	fromSquare := rand.Intn(64)
	toSquare := rand.Intn(64)
	moveLabel := fromSquare*64 + toSquare // Simple encoding: 0-4095

	return chessSample{
		state: state,
		move:  moveLabel,
	}
}

// displayStats shows current store statistics
func displayStats(store *storage.ObservationStore) {
	stats, err := store.GetStats()
	if err != nil {
		fmt.Fprintf(os.Stderr, "❌ Failed to get stats: %v\n", err)
		return
	}

	fmt.Println("Current Statistics:")
	fmt.Println("───────────────────────────────────────────────────────────")
	fmt.Printf("  Total Samples:  %d\n", stats.TotalSamples)
	fmt.Printf("  Actual Samples: %d\n", stats.ActualSamples)
	fmt.Printf("  Max Capacity:   %d\n", stats.MaxSize)
	fmt.Printf("  Buffer Wrapped: %v\n", stats.IsWrapped)
	fmt.Printf("  Database Path:  %s\n", stats.DBPath)

	if stats.ActualSamples > 0 {
		utilization := float64(stats.ActualSamples) / float64(stats.MaxSize) * 100
		fmt.Printf("  Utilization:    %.1f%%\n", utilization)
	}
	fmt.Println("───────────────────────────────────────────────────────────")
}

// testRetrieval tests batch retrieval functionality
func testRetrieval(store *storage.ObservationStore, batchSize int) {
	count, err := store.CountSamples()
	if err != nil {
		fmt.Fprintf(os.Stderr, "❌ Failed to count samples: %v\n", err)
		return
	}

	if count == 0 {
		fmt.Println("⚠️  No samples to retrieve")
		return
	}

	fmt.Printf("Testing batch retrieval (batch size: %d)...\n", batchSize)

	// Test random batch
	startTime := time.Now()
	samples, err := store.GetBatch(batchSize)
	if err != nil {
		fmt.Fprintf(os.Stderr, "❌ Failed to retrieve batch: %v\n", err)
		return
	}
	duration := time.Since(startTime)

	fmt.Printf("✓ Retrieved %d samples in %v\n", len(samples), duration)

	// Analyze retrieved samples
	fmt.Println("\nSample Analysis:")
	fmt.Println("───────────────────────────────────────────────────────────")

	// Calculate statistics
	var totalPieces float64
	var minMove, maxMove int = 4096, 0
	var moveHistogram = make(map[int]int)

	for _, sample := range samples {
		// Count pieces on board
		pieces := 0
		for _, val := range sample.State {
			if val > 0.01 { // Non-empty square
				pieces++
			}
		}
		totalPieces += float64(pieces)

		// Track move range
		if sample.MoveLabel < minMove {
			minMove = sample.MoveLabel
		}
		if sample.MoveLabel > maxMove {
			maxMove = sample.MoveLabel
		}

		// Build histogram of moves
		moveHistogram[sample.MoveLabel]++
	}

	avgPieces := totalPieces / float64(len(samples))
	fmt.Printf("  Avg pieces per board: %.1f\n", avgPieces)
	fmt.Printf("  Move range: %d - %d\n", minMove, maxMove)
	fmt.Printf("  Unique moves: %d\n", len(moveHistogram))

	// Show first 3 samples as examples
	fmt.Println("\nFirst 3 Sample States:")
	for i := 0; i < 3 && i < len(samples); i++ {
		sample := samples[i]
		fmt.Printf("\n  Sample %d:\n", i+1)
		fmt.Printf("    Move: %d (from=%d, to=%d)\n",
			sample.MoveLabel, sample.MoveLabel/64, sample.MoveLabel%64)
		fmt.Printf("    Timestamp: %s\n", time.Unix(sample.Timestamp, 0).Format("15:04:05"))

		// Show board visualization (simplified)
		fmt.Printf("    Board (first 16 squares):\n      ")
		for j := 0; j < 16 && j < len(sample.State); j++ {
			if sample.State[j] < 0.01 {
				fmt.Print("· ")
			} else if sample.State[j] < 0.5 {
				fmt.Print("♙ ") // White piece
			} else {
				fmt.Print("♟ ") // Black piece
			}
			if (j+1)%8 == 0 {
				fmt.Print("\n      ")
			}
		}
		fmt.Println()
	}

	fmt.Println("───────────────────────────────────────────────────────────")

	// Test sequential retrieval
	fmt.Println("\nTesting sequential batch retrieval...")
	seqSamples, err := store.GetSequentialBatch(5, 0)
	if err != nil {
		fmt.Fprintf(os.Stderr, "❌ Failed to retrieve sequential batch: %v\n", err)
		return
	}
	fmt.Printf("✓ Retrieved %d samples sequentially\n", len(seqSamples))
}

func init() {
	rand.Seed(time.Now().UnixNano())
}
