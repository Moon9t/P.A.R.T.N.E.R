package data

import (
	"fmt"
	"path/filepath"
	"sync"
	"sync/atomic"
)

// IngestionConfig holds configuration for PGN ingestion
type IngestionConfig struct {
	PGNPath        string // Path to PGN file or directory
	DatasetPath    string // Path to output dataset
	MaxGames       int    // Maximum number of games to process (0 = all)
	MaxPositions   int    // Maximum positions to extract (0 = all)
	SkipInvalid    bool   // Skip invalid positions instead of failing
	BatchSize      int    // Number of entries to batch before writing
	Verbose        bool   // Print progress information
	WorkerPoolSize int    // Number of parallel workers (0 = sequential)
}

// DefaultIngestionConfig returns a config with sensible defaults
func DefaultIngestionConfig(pgnPath, datasetPath string) *IngestionConfig {
	return &IngestionConfig{
		PGNPath:        pgnPath,
		DatasetPath:    datasetPath,
		MaxGames:       0,
		MaxPositions:   0,
		SkipInvalid:    true,
		BatchSize:      100,
		Verbose:        true,
		WorkerPoolSize: 4,
	}
}

// Ingestor handles the ingestion of PGN files into a dataset
type Ingestor struct {
	config  *IngestionConfig
	dataset *Dataset
}

// NewIngestor creates a new ingestor
func NewIngestor(config *IngestionConfig) (*Ingestor, error) {
	dataset, err := NewDataset(config.DatasetPath)
	if err != nil {
		return nil, fmt.Errorf("failed to create dataset: %w", err)
	}

	return &Ingestor{
		config:  config,
		dataset: dataset,
	}, nil
}

// Close closes the ingestor and underlying dataset
func (ing *Ingestor) Close() error {
	return ing.dataset.Close()
}

// Ingest processes PGN files and populates the dataset
func (ing *Ingestor) Ingest() (*IngestionStats, error) {
	stats := &IngestionStats{}

	// Parse PGN file
	parser := NewPGNParser(ing.config.PGNPath)
	games, err := parser.ParsePGN()
	if err != nil {
		return stats, fmt.Errorf("failed to parse PGN: %w", err)
	}

	stats.TotalGames = len(games)

	if ing.config.Verbose {
		fmt.Printf("Parsed %d games from %s\n", len(games), filepath.Base(ing.config.PGNPath))
	}

	// Limit games if configured
	if ing.config.MaxGames > 0 && len(games) > ing.config.MaxGames {
		games = games[:ing.config.MaxGames]
		if ing.config.Verbose {
			fmt.Printf("Limited to %d games\n", ing.config.MaxGames)
		}
	}

	// Process games in batches
	var batchMu sync.Mutex
	var currentBatch []*DataEntry
	var positionsProcessed int32

	// Worker function
	processGame := func(gameIdx int) error {
		game := games[gameIdx]
		gameID := fmt.Sprintf("game_%d", gameIdx)

		positions, err := ExtractPositions(game)
		if err != nil {
			if ing.config.SkipInvalid {
				atomic.AddInt32(&stats.SkippedPositions, 1)
				return nil
			}
			return fmt.Errorf("failed to extract positions from game %d: %w", gameIdx, err)
		}

		for moveNum, pos := range positions {
			// Check max positions limit
			if ing.config.MaxPositions > 0 && int(atomic.LoadInt32(&positionsProcessed)) >= ing.config.MaxPositions {
				break
			}

			// Tensorize board
			tensor, err := TensorizeBoard(pos.Board)
			if err != nil {
				if ing.config.SkipInvalid {
					atomic.AddInt32(&stats.SkippedPositions, 1)
					continue
				}
				return fmt.Errorf("failed to tensorize board: %w", err)
			}

			// Encode move
			fromSquare, toSquare, err := EncodeMoveLabel(pos.Move)
			if err != nil {
				if ing.config.SkipInvalid {
					atomic.AddInt32(&stats.SkippedPositions, 1)
					continue
				}
				return fmt.Errorf("failed to encode move: %w", err)
			}

			// Create entry
			entry := &DataEntry{
				StateTensor: TensorToFlatArray(tensor),
				FromSquare:  fromSquare,
				ToSquare:    toSquare,
				GameID:      gameID,
				MoveNumber:  moveNum,
			}

			// Add to batch
			batchMu.Lock()
			currentBatch = append(currentBatch, entry)
			batchLen := len(currentBatch)
			batchMu.Unlock()

			atomic.AddInt32(&positionsProcessed, 1)

			// Write batch if full
			if batchLen >= ing.config.BatchSize {
				batchMu.Lock()
				if len(currentBatch) >= ing.config.BatchSize {
					toWrite := currentBatch[:ing.config.BatchSize]
					currentBatch = currentBatch[ing.config.BatchSize:]
					batchMu.Unlock()

					if err := ing.dataset.AddBatch(toWrite); err != nil {
						return fmt.Errorf("failed to add batch: %w", err)
					}
					atomic.AddInt32(&stats.PositionsIngested, int32(len(toWrite)))

					if ing.config.Verbose && atomic.LoadInt32(&stats.PositionsIngested)%1000 == 0 {
						fmt.Printf("Ingested %d positions...\n", atomic.LoadInt32(&stats.PositionsIngested))
					}
				} else {
					batchMu.Unlock()
				}
			}
		}

		atomic.AddInt32(&stats.GamesProcessed, 1)
		return nil
	}

	// Process games (sequential or parallel)
	if ing.config.WorkerPoolSize <= 1 {
		// Sequential processing
		for i := range games {
			if err := processGame(i); err != nil {
				return stats, err
			}
		}
	} else {
		// Parallel processing
		var wg sync.WaitGroup
		semaphore := make(chan struct{}, ing.config.WorkerPoolSize)
		errChan := make(chan error, 1)

		for i := range games {
			wg.Add(1)
			go func(idx int) {
				defer wg.Done()

				semaphore <- struct{}{}
				defer func() { <-semaphore }()

				if err := processGame(idx); err != nil {
					select {
					case errChan <- err:
					default:
					}
				}
			}(i)
		}

		wg.Wait()
		close(errChan)

		if err := <-errChan; err != nil {
			return stats, err
		}
	}

	// Write remaining batch
	batchMu.Lock()
	if len(currentBatch) > 0 {
		if err := ing.dataset.AddBatch(currentBatch); err != nil {
			batchMu.Unlock()
			return stats, fmt.Errorf("failed to write final batch: %w", err)
		}
		atomic.AddInt32(&stats.PositionsIngested, int32(len(currentBatch)))
	}
	batchMu.Unlock()

	stats.PositionsIngested = positionsProcessed

	if ing.config.Verbose {
		fmt.Printf("\nIngestion complete:\n")
		fmt.Printf("  Games processed: %d/%d\n", stats.GamesProcessed, stats.TotalGames)
		fmt.Printf("  Positions ingested: %d\n", stats.PositionsIngested)
		fmt.Printf("  Positions skipped: %d\n", stats.SkippedPositions)
	}

	return stats, nil
}

// IngestionStats contains statistics about the ingestion process
type IngestionStats struct {
	TotalGames        int
	GamesProcessed    int32
	PositionsIngested int32
	SkippedPositions  int32
}
