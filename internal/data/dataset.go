package data

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sync"

	bolt "go.etcd.io/bbolt"
)

const (
	// DefaultBucketName is the default bucket name for chess positions
	DefaultBucketName = "chess_positions"
)

// DataEntry represents a single training example
type DataEntry struct {
	StateTensor []float32 `json:"state_tensor"` // Flat array of [12][8][8] tensor
	FromSquare  int       `json:"from_square"`  // Move from square (0-63)
	ToSquare    int       `json:"to_square"`    // Move to square (0-63)
	GameID      string    `json:"game_id"`      // Optional: game identifier
	MoveNumber  int       `json:"move_number"`  // Optional: move number in game
}

// Dataset manages the on-disk chess dataset using BoltDB
type Dataset struct {
	db         *bolt.DB
	bucketName string
	path       string
	mu         sync.RWMutex
}

// NewDataset creates a new dataset or opens an existing one
func NewDataset(path string) (*Dataset, error) {
	// Ensure directory exists
	dir := filepath.Dir(path)
	if err := os.MkdirAll(dir, 0755); err != nil {
		return nil, fmt.Errorf("failed to create directory: %w", err)
	}

	db, err := bolt.Open(path, 0600, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to open database: %w", err)
	}

	ds := &Dataset{
		db:         db,
		bucketName: DefaultBucketName,
		path:       path,
	}

	// Create bucket if it doesn't exist
	err = db.Update(func(tx *bolt.Tx) error {
		_, err := tx.CreateBucketIfNotExists([]byte(ds.bucketName))
		return err
	})
	if err != nil {
		db.Close()
		return nil, fmt.Errorf("failed to create bucket: %w", err)
	}

	return ds, nil
}

// Close closes the dataset
func (ds *Dataset) Close() error {
	ds.mu.Lock()
	defer ds.mu.Unlock()

	if ds.db != nil {
		return ds.db.Close()
	}
	return nil
}

// Add adds a new entry to the dataset
func (ds *Dataset) Add(entry *DataEntry) error {
	ds.mu.Lock()
	defer ds.mu.Unlock()

	return ds.db.Update(func(tx *bolt.Tx) error {
		bucket := tx.Bucket([]byte(ds.bucketName))
		if bucket == nil {
			return fmt.Errorf("bucket not found")
		}

		// Get next ID
		id, _ := bucket.NextSequence()
		key := []byte(fmt.Sprintf("%020d", id))

		// Serialize entry
		value, err := json.Marshal(entry)
		if err != nil {
			return fmt.Errorf("failed to marshal entry: %w", err)
		}

		return bucket.Put(key, value)
	})
}

// AddBatch adds multiple entries in a single transaction
func (ds *Dataset) AddBatch(entries []*DataEntry) error {
	ds.mu.Lock()
	defer ds.mu.Unlock()

	return ds.db.Update(func(tx *bolt.Tx) error {
		bucket := tx.Bucket([]byte(ds.bucketName))
		if bucket == nil {
			return fmt.Errorf("bucket not found")
		}

		for _, entry := range entries {
			id, _ := bucket.NextSequence()
			key := []byte(fmt.Sprintf("%020d", id))

			value, err := json.Marshal(entry)
			if err != nil {
				return fmt.Errorf("failed to marshal entry: %w", err)
			}

			if err := bucket.Put(key, value); err != nil {
				return err
			}
		}

		return nil
	})
}

// Count returns the number of entries in the dataset
func (ds *Dataset) Count() (int, error) {
	ds.mu.RLock()
	defer ds.mu.RUnlock()

	count := 0
	err := ds.db.View(func(tx *bolt.Tx) error {
		bucket := tx.Bucket([]byte(ds.bucketName))
		if bucket == nil {
			return fmt.Errorf("bucket not found")
		}

		bucket.ForEach(func(k, v []byte) error {
			count++
			return nil
		})

		return nil
	})

	return count, err
}

// LoadBatch loads n entries starting from the given offset
// This allows streaming access without loading everything into memory
func (ds *Dataset) LoadBatch(offset, n int) ([]*DataEntry, error) {
	ds.mu.RLock()
	defer ds.mu.RUnlock()

	var entries []*DataEntry

	err := ds.db.View(func(tx *bolt.Tx) error {
		bucket := tx.Bucket([]byte(ds.bucketName))
		if bucket == nil {
			return fmt.Errorf("bucket not found")
		}

		cursor := bucket.Cursor()

		// Skip to offset
		currentIdx := 0
		k, v := cursor.First()
		for k != nil && currentIdx < offset {
			k, v = cursor.Next()
			currentIdx++
		}

		// Load n entries
		loaded := 0
		for k != nil && loaded < n {
			var entry DataEntry
			if err := json.Unmarshal(v, &entry); err != nil {
				return fmt.Errorf("failed to unmarshal entry: %w", err)
			}

			entries = append(entries, &entry)
			loaded++

			k, v = cursor.Next()
		}

		return nil
	})

	return entries, err
}

// LoadAll loads all entries (use with caution for large datasets)
func (ds *Dataset) LoadAll() ([]*DataEntry, error) {
	count, err := ds.Count()
	if err != nil {
		return nil, err
	}

	return ds.LoadBatch(0, count)
}

// VerifyIntegrity scans the dataset and verifies all tensors are valid
func (ds *Dataset) VerifyIntegrity() error {
	ds.mu.RLock()
	defer ds.mu.RUnlock()

	errors := 0
	totalEntries := 0

	err := ds.db.View(func(tx *bolt.Tx) error {
		bucket := tx.Bucket([]byte(ds.bucketName))
		if bucket == nil {
			return fmt.Errorf("bucket not found")
		}

		return bucket.ForEach(func(k, v []byte) error {
			totalEntries++

			var entry DataEntry
			if err := json.Unmarshal(v, &entry); err != nil {
				errors++
				return nil // Continue checking other entries
			}

			// Verify tensor dimensions
			expectedLen := NumChannels * BoardSize * BoardSize
			if len(entry.StateTensor) != expectedLen {
				errors++
				return nil
			}

			// Verify move labels
			if entry.FromSquare < 0 || entry.FromSquare >= 64 {
				errors++
				return nil
			}
			if entry.ToSquare < 0 || entry.ToSquare >= 64 {
				errors++
				return nil
			}

			// Reconstruct tensor and validate
			tensor, err := FlatArrayToTensor(entry.StateTensor)
			if err != nil {
				errors++
				return nil
			}

			if err := ValidateTensor(tensor); err != nil {
				errors++
				return nil
			}

			return nil
		})
	})

	if err != nil {
		return err
	}

	if errors > 0 {
		return fmt.Errorf("integrity check failed: %d/%d entries have errors", errors, totalEntries)
	}

	return nil
}

// Clear removes all entries from the dataset
func (ds *Dataset) Clear() error {
	ds.mu.Lock()
	defer ds.mu.Unlock()

	return ds.db.Update(func(tx *bolt.Tx) error {
		if err := tx.DeleteBucket([]byte(ds.bucketName)); err != nil {
			return err
		}
		_, err := tx.CreateBucket([]byte(ds.bucketName))
		return err
	})
}

// GetStats returns statistics about the dataset
func (ds *Dataset) GetStats() (*DatasetStats, error) {
	count, err := ds.Count()
	if err != nil {
		return nil, err
	}

	fileInfo, err := os.Stat(ds.path)
	if err != nil {
		return nil, err
	}

	return &DatasetStats{
		TotalEntries: count,
		FilePath:     ds.path,
		FileSize:     fileInfo.Size(),
	}, nil
}

// DatasetStats contains statistics about the dataset
type DatasetStats struct {
	TotalEntries int
	FilePath     string
	FileSize     int64
}
