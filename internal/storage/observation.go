package storage

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math/rand"
	"time"

	"go.etcd.io/bbolt"
)

const (
	// BucketName for storing observations
	BucketName = "observations"

	// MetaBucket for storing metadata
	MetaBucket = "meta"

	// CountKey for tracking total samples
	CountKey = "count"
)

// Sample represents a single observation with state and move
type Sample struct {
	State     []float64 `json:"state"`      // Board state tensor (64 values for 8x8)
	MoveLabel int       `json:"move_label"` // Move index (0-4095 for chess)
	Timestamp int64     `json:"timestamp"`  // Unix timestamp
}

// ObservationStore manages the storage of observation samples
type ObservationStore struct {
	db       *bbolt.DB
	dbPath   string
	maxSize  int
	count    uint64
	isClosed bool
}

// NewObservationStore creates a new observation store with BoltDB backend
func NewObservationStore(dbPath string, maxSize int) (*ObservationStore, error) {
	// Open database with timeout
	db, err := bbolt.Open(dbPath, 0600, &bbolt.Options{
		Timeout: 5 * time.Second,
	})
	if err != nil {
		return nil, fmt.Errorf("failed to open database: %w", err)
	}

	// Initialize buckets
	err = db.Update(func(tx *bbolt.Tx) error {
		_, err := tx.CreateBucketIfNotExists([]byte(BucketName))
		if err != nil {
			return fmt.Errorf("create bucket: %w", err)
		}

		_, err = tx.CreateBucketIfNotExists([]byte(MetaBucket))
		if err != nil {
			return fmt.Errorf("create meta bucket: %w", err)
		}

		return nil
	})
	if err != nil {
		db.Close()
		return nil, err
	}

	store := &ObservationStore{
		db:      db,
		dbPath:  dbPath,
		maxSize: maxSize,
	}

	// Load current count
	count, err := store.CountSamples()
	if err != nil {
		db.Close()
		return nil, err
	}
	store.count = count

	return store, nil
}

// StoreSample stores a new observation sample
func (s *ObservationStore) StoreSample(stateTensor []float64, moveLabel int) error {
	if s.isClosed {
		return fmt.Errorf("store is closed")
	}

	// Validate input
	if len(stateTensor) != 64 {
		return fmt.Errorf("invalid state tensor size: expected 64, got %d", len(stateTensor))
	}
	if moveLabel < 0 || moveLabel >= 4096 {
		return fmt.Errorf("invalid move label: %d (must be 0-4095)", moveLabel)
	}

	sample := Sample{
		State:     stateTensor,
		MoveLabel: moveLabel,
		Timestamp: time.Now().Unix(),
	}

	// Serialize sample
	data, err := json.Marshal(sample)
	if err != nil {
		return fmt.Errorf("failed to marshal sample: %w", err)
	}

	// Store in database
	err = s.db.Update(func(tx *bbolt.Tx) error {
		b := tx.Bucket([]byte(BucketName))
		if b == nil {
			return fmt.Errorf("bucket not found")
		}

		// Use current count as key (implements circular buffer)
		key := s.count % uint64(s.maxSize)
		keyBytes := make([]byte, 8)
		binary.BigEndian.PutUint64(keyBytes, key)

		// Store sample
		if err := b.Put(keyBytes, data); err != nil {
			return err
		}

		// Update count in meta bucket
		meta := tx.Bucket([]byte(MetaBucket))
		if meta == nil {
			return fmt.Errorf("meta bucket not found")
		}

		s.count++
		countBytes := make([]byte, 8)
		binary.BigEndian.PutUint64(countBytes, s.count)

		return meta.Put([]byte(CountKey), countBytes)
	})

	return err
}

// GetBatch retrieves a batch of random samples for training
func (s *ObservationStore) GetBatch(size int) ([]Sample, error) {
	if s.isClosed {
		return nil, fmt.Errorf("store is closed")
	}

	count, err := s.CountSamples()
	if err != nil {
		return nil, err
	}

	if count == 0 {
		return nil, fmt.Errorf("no samples available")
	}

	// Limit batch size to available samples
	actualSize := size
	if uint64(size) > count {
		actualSize = int(count)
	}

	// Generate random indices
	indices := make(map[uint64]bool)
	for len(indices) < actualSize {
		idx := uint64(rand.Intn(int(count)))
		if count > uint64(s.maxSize) {
			// If we've wrapped around, only sample from valid range
			idx = idx % uint64(s.maxSize)
		}
		indices[idx] = true
	}

	// Fetch samples
	samples := make([]Sample, 0, actualSize)

	err = s.db.View(func(tx *bbolt.Tx) error {
		b := tx.Bucket([]byte(BucketName))
		if b == nil {
			return fmt.Errorf("bucket not found")
		}

		for idx := range indices {
			keyBytes := make([]byte, 8)
			binary.BigEndian.PutUint64(keyBytes, idx)

			data := b.Get(keyBytes)
			if data == nil {
				continue // Skip missing samples
			}

			var sample Sample
			if err := json.Unmarshal(data, &sample); err != nil {
				continue // Skip corrupted samples
			}

			samples = append(samples, sample)
		}

		return nil
	})

	if err != nil {
		return nil, err
	}

	if len(samples) == 0 {
		return nil, fmt.Errorf("no valid samples retrieved")
	}

	return samples, nil
}

// GetSequentialBatch retrieves a sequential batch of samples (for deterministic testing)
func (s *ObservationStore) GetSequentialBatch(size int, offset int) ([]Sample, error) {
	if s.isClosed {
		return nil, fmt.Errorf("store is closed")
	}

	count, err := s.CountSamples()
	if err != nil {
		return nil, err
	}

	if count == 0 {
		return nil, fmt.Errorf("no samples available")
	}

	// Calculate actual range
	actualSize := size
	if uint64(offset+size) > count {
		if uint64(offset) >= count {
			return nil, fmt.Errorf("offset exceeds sample count")
		}
		actualSize = int(count) - offset
	}

	samples := make([]Sample, 0, actualSize)

	err = s.db.View(func(tx *bbolt.Tx) error {
		b := tx.Bucket([]byte(BucketName))
		if b == nil {
			return fmt.Errorf("bucket not found")
		}

		for i := 0; i < actualSize; i++ {
			idx := uint64(offset + i)
			if count > uint64(s.maxSize) {
				idx = idx % uint64(s.maxSize)
			}

			keyBytes := make([]byte, 8)
			binary.BigEndian.PutUint64(keyBytes, idx)

			data := b.Get(keyBytes)
			if data == nil {
				continue
			}

			var sample Sample
			if err := json.Unmarshal(data, &sample); err != nil {
				continue
			}

			samples = append(samples, sample)
		}

		return nil
	})

	return samples, err
}

// CountSamples returns the total number of samples stored
func (s *ObservationStore) CountSamples() (uint64, error) {
	if s.isClosed {
		return 0, fmt.Errorf("store is closed")
	}

	var count uint64

	err := s.db.View(func(tx *bbolt.Tx) error {
		meta := tx.Bucket([]byte(MetaBucket))
		if meta == nil {
			return fmt.Errorf("meta bucket not found")
		}

		countBytes := meta.Get([]byte(CountKey))
		if countBytes == nil {
			count = 0
			return nil
		}

		count = binary.BigEndian.Uint64(countBytes)
		return nil
	})

	return count, err
}

// GetActualSize returns the actual number of unique samples (handles circular buffer)
func (s *ObservationStore) GetActualSize() (int, error) {
	count, err := s.CountSamples()
	if err != nil {
		return 0, err
	}

	if count > uint64(s.maxSize) {
		return s.maxSize, nil
	}

	return int(count), nil
}

// Clear removes all samples from the store
func (s *ObservationStore) Clear() error {
	if s.isClosed {
		return fmt.Errorf("store is closed")
	}

	return s.db.Update(func(tx *bbolt.Tx) error {
		// Delete and recreate bucket
		if err := tx.DeleteBucket([]byte(BucketName)); err != nil {
			return err
		}

		if _, err := tx.CreateBucket([]byte(BucketName)); err != nil {
			return err
		}

		// Reset count
		meta := tx.Bucket([]byte(MetaBucket))
		if meta == nil {
			return fmt.Errorf("meta bucket not found")
		}

		s.count = 0
		countBytes := make([]byte, 8)
		binary.BigEndian.PutUint64(countBytes, 0)

		return meta.Put([]byte(CountKey), countBytes)
	})
}

// Close closes the database connection
func (s *ObservationStore) Close() error {
	if s.isClosed {
		return nil
	}

	s.isClosed = true
	return s.db.Close()
}

// Stats returns statistics about the store
type Stats struct {
	TotalSamples  uint64
	ActualSamples int
	MaxSize       int
	DBPath        string
	IsWrapped     bool
}

// GetStats returns current statistics
func (s *ObservationStore) GetStats() (Stats, error) {
	count, err := s.CountSamples()
	if err != nil {
		return Stats{}, err
	}

	actualSize, err := s.GetActualSize()
	if err != nil {
		return Stats{}, err
	}

	return Stats{
		TotalSamples:  count,
		ActualSamples: actualSize,
		MaxSize:       s.maxSize,
		DBPath:        s.dbPath,
		IsWrapped:     count > uint64(s.maxSize),
	}, nil
}

// ExportToJSON exports all samples to a JSON file (for backup/debugging)
func (s *ObservationStore) ExportToJSON(outputPath string) error {
	if s.isClosed {
		return fmt.Errorf("store is closed")
	}

	actualSize, err := s.GetActualSize()
	if err != nil {
		return err
	}

	samples, err := s.GetSequentialBatch(actualSize, 0)
	if err != nil {
		return err
	}

	data, err := json.MarshalIndent(samples, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to marshal samples: %w", err)
	}

	// Write to file (using os package would be imported here)
	return fmt.Errorf("export not yet implemented - would write %d bytes to %s", len(data), outputPath)
}
