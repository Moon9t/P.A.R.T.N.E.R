package storage

import (
	"math/rand"
	"path/filepath"
	"testing"
	"time"
)

// TestNewObservationStore tests store creation
func TestNewObservationStore(t *testing.T) {
	tmpDir := t.TempDir()
	dbPath := filepath.Join(tmpDir, "test.db")

	store, err := NewObservationStore(dbPath, 1000)
	if err != nil {
		t.Fatalf("Failed to create store: %v", err)
	}
	defer store.Close()

	if store.dbPath != dbPath {
		t.Errorf("Expected dbPath %s, got %s", dbPath, store.dbPath)
	}

	if store.maxSize != 1000 {
		t.Errorf("Expected maxSize 1000, got %d", store.maxSize)
	}

	// Verify count is 0 initially
	count, err := store.CountSamples()
	if err != nil {
		t.Fatalf("Failed to count samples: %v", err)
	}

	if count != 0 {
		t.Errorf("Expected initial count 0, got %d", count)
	}
}

// TestStoreSample tests storing a single sample
func TestStoreSample(t *testing.T) {
	tmpDir := t.TempDir()
	dbPath := filepath.Join(tmpDir, "test.db")

	store, err := NewObservationStore(dbPath, 1000)
	if err != nil {
		t.Fatalf("Failed to create store: %v", err)
	}
	defer store.Close()

	// Create test sample
	state := make([]float64, 64)
	for i := range state {
		state[i] = rand.Float64()
	}
	moveLabel := 42

	// Store sample
	err = store.StoreSample(state, moveLabel)
	if err != nil {
		t.Fatalf("Failed to store sample: %v", err)
	}

	// Verify count increased
	count, err := store.CountSamples()
	if err != nil {
		t.Fatalf("Failed to count samples: %v", err)
	}

	if count != 1 {
		t.Errorf("Expected count 1, got %d", count)
	}
}

// TestStoreSampleValidation tests input validation
func TestStoreSampleValidation(t *testing.T) {
	tmpDir := t.TempDir()
	dbPath := filepath.Join(tmpDir, "test.db")

	store, err := NewObservationStore(dbPath, 1000)
	if err != nil {
		t.Fatalf("Failed to create store: %v", err)
	}
	defer store.Close()

	// Test invalid state size
	invalidState := make([]float64, 32) // Wrong size
	err = store.StoreSample(invalidState, 42)
	if err == nil {
		t.Error("Expected error for invalid state size, got nil")
	}

	// Test invalid move label (negative)
	validState := make([]float64, 64)
	err = store.StoreSample(validState, -1)
	if err == nil {
		t.Error("Expected error for negative move label, got nil")
	}

	// Test invalid move label (too large)
	err = store.StoreSample(validState, 5000)
	if err == nil {
		t.Error("Expected error for move label > 4095, got nil")
	}

	// Test valid input
	err = store.StoreSample(validState, 100)
	if err != nil {
		t.Errorf("Expected no error for valid input, got %v", err)
	}
}

// TestGetBatch tests batch retrieval
func TestGetBatch(t *testing.T) {
	tmpDir := t.TempDir()
	dbPath := filepath.Join(tmpDir, "test.db")

	store, err := NewObservationStore(dbPath, 1000)
	if err != nil {
		t.Fatalf("Failed to create store: %v", err)
	}
	defer store.Close()

	// Store multiple samples
	numSamples := 50
	for i := 0; i < numSamples; i++ {
		state := make([]float64, 64)
		for j := range state {
			state[j] = float64(i) + rand.Float64()
		}

		err = store.StoreSample(state, i%4096)
		if err != nil {
			t.Fatalf("Failed to store sample %d: %v", i, err)
		}
	}

	// Test batch retrieval
	batchSize := 10
	samples, err := store.GetBatch(batchSize)
	if err != nil {
		t.Fatalf("Failed to get batch: %v", err)
	}

	if len(samples) != batchSize {
		t.Errorf("Expected batch size %d, got %d", batchSize, len(samples))
	}

	// Verify samples are valid
	for i, sample := range samples {
		if len(sample.State) != 64 {
			t.Errorf("Sample %d has invalid state size: %d", i, len(sample.State))
		}
		if sample.MoveLabel < 0 || sample.MoveLabel >= 4096 {
			t.Errorf("Sample %d has invalid move label: %d", i, sample.MoveLabel)
		}
		if sample.Timestamp == 0 {
			t.Errorf("Sample %d has zero timestamp", i)
		}
	}
}

// TestGetBatchWithLimitedSamples tests batch retrieval with fewer samples than requested
func TestGetBatchWithLimitedSamples(t *testing.T) {
	tmpDir := t.TempDir()
	dbPath := filepath.Join(tmpDir, "test.db")

	store, err := NewObservationStore(dbPath, 1000)
	if err != nil {
		t.Fatalf("Failed to create store: %v", err)
	}
	defer store.Close()

	// Store only 5 samples
	numSamples := 5
	for i := 0; i < numSamples; i++ {
		state := make([]float64, 64)
		err = store.StoreSample(state, i)
		if err != nil {
			t.Fatalf("Failed to store sample: %v", err)
		}
	}

	// Request more samples than available
	samples, err := store.GetBatch(20)
	if err != nil {
		t.Fatalf("Failed to get batch: %v", err)
	}

	// Should return all available samples
	if len(samples) != numSamples {
		t.Errorf("Expected %d samples, got %d", numSamples, len(samples))
	}
}

// TestGetSequentialBatch tests sequential batch retrieval
func TestGetSequentialBatch(t *testing.T) {
	tmpDir := t.TempDir()
	dbPath := filepath.Join(tmpDir, "test.db")

	store, err := NewObservationStore(dbPath, 1000)
	if err != nil {
		t.Fatalf("Failed to create store: %v", err)
	}
	defer store.Close()

	// Store samples with predictable data
	numSamples := 20
	for i := 0; i < numSamples; i++ {
		state := make([]float64, 64)
		state[0] = float64(i) // Use first element as identifier

		err = store.StoreSample(state, i)
		if err != nil {
			t.Fatalf("Failed to store sample: %v", err)
		}
	}

	// Get sequential batch
	samples, err := store.GetSequentialBatch(5, 10)
	if err != nil {
		t.Fatalf("Failed to get sequential batch: %v", err)
	}

	if len(samples) != 5 {
		t.Errorf("Expected 5 samples, got %d", len(samples))
	}

	// Verify sequential order
	for i, sample := range samples {
		expected := float64(10 + i)
		if sample.State[0] != expected {
			t.Errorf("Sample %d: expected state[0]=%f, got %f", i, expected, sample.State[0])
		}
	}
}

// TestCircularBuffer tests that the store implements a circular buffer
func TestCircularBuffer(t *testing.T) {
	tmpDir := t.TempDir()
	dbPath := filepath.Join(tmpDir, "test.db")

	// Create store with small max size
	maxSize := 10
	store, err := NewObservationStore(dbPath, maxSize)
	if err != nil {
		t.Fatalf("Failed to create store: %v", err)
	}
	defer store.Close()

	// Store more samples than max size
	numSamples := 25
	for i := 0; i < numSamples; i++ {
		state := make([]float64, 64)
		state[0] = float64(i)

		err = store.StoreSample(state, i)
		if err != nil {
			t.Fatalf("Failed to store sample: %v", err)
		}
	}

	// Total count should be numSamples
	count, err := store.CountSamples()
	if err != nil {
		t.Fatalf("Failed to count samples: %v", err)
	}
	if count != uint64(numSamples) {
		t.Errorf("Expected count %d, got %d", numSamples, count)
	}

	// Actual size should be capped at maxSize
	actualSize, err := store.GetActualSize()
	if err != nil {
		t.Fatalf("Failed to get actual size: %v", err)
	}
	if actualSize != maxSize {
		t.Errorf("Expected actual size %d, got %d", maxSize, actualSize)
	}

	// Stats should show wrapped
	stats, err := store.GetStats()
	if err != nil {
		t.Fatalf("Failed to get stats: %v", err)
	}
	if !stats.IsWrapped {
		t.Error("Expected IsWrapped to be true")
	}
}

// TestClear tests clearing all samples
func TestClear(t *testing.T) {
	tmpDir := t.TempDir()
	dbPath := filepath.Join(tmpDir, "test.db")

	store, err := NewObservationStore(dbPath, 1000)
	if err != nil {
		t.Fatalf("Failed to create store: %v", err)
	}
	defer store.Close()

	// Store some samples
	for i := 0; i < 10; i++ {
		state := make([]float64, 64)
		err = store.StoreSample(state, i)
		if err != nil {
			t.Fatalf("Failed to store sample: %v", err)
		}
	}

	// Clear
	err = store.Clear()
	if err != nil {
		t.Fatalf("Failed to clear: %v", err)
	}

	// Verify count is 0
	count, err := store.CountSamples()
	if err != nil {
		t.Fatalf("Failed to count samples: %v", err)
	}
	if count != 0 {
		t.Errorf("Expected count 0 after clear, got %d", count)
	}
}

// TestPersistence tests that data persists across store instances
func TestPersistence(t *testing.T) {
	tmpDir := t.TempDir()
	dbPath := filepath.Join(tmpDir, "test.db")

	// Create store and add samples
	{
		store, err := NewObservationStore(dbPath, 1000)
		if err != nil {
			t.Fatalf("Failed to create store: %v", err)
		}

		for i := 0; i < 5; i++ {
			state := make([]float64, 64)
			state[0] = float64(i)
			err = store.StoreSample(state, i)
			if err != nil {
				t.Fatalf("Failed to store sample: %v", err)
			}
		}

		store.Close()
	}

	// Reopen and verify data persists
	{
		store, err := NewObservationStore(dbPath, 1000)
		if err != nil {
			t.Fatalf("Failed to reopen store: %v", err)
		}
		defer store.Close()

		count, err := store.CountSamples()
		if err != nil {
			t.Fatalf("Failed to count samples: %v", err)
		}

		if count != 5 {
			t.Errorf("Expected count 5 after reopen, got %d", count)
		}

		// Verify data integrity
		samples, err := store.GetSequentialBatch(5, 0)
		if err != nil {
			t.Fatalf("Failed to get samples: %v", err)
		}

		for i, sample := range samples {
			if sample.State[0] != float64(i) {
				t.Errorf("Sample %d: expected state[0]=%f, got %f", i, float64(i), sample.State[0])
			}
		}
	}
}

// TestGetStats tests statistics retrieval
func TestGetStats(t *testing.T) {
	tmpDir := t.TempDir()
	dbPath := filepath.Join(tmpDir, "test.db")

	store, err := NewObservationStore(dbPath, 100)
	if err != nil {
		t.Fatalf("Failed to create store: %v", err)
	}
	defer store.Close()

	// Store samples
	for i := 0; i < 25; i++ {
		state := make([]float64, 64)
		err = store.StoreSample(state, i)
		if err != nil {
			t.Fatalf("Failed to store sample: %v", err)
		}
	}

	stats, err := store.GetStats()
	if err != nil {
		t.Fatalf("Failed to get stats: %v", err)
	}

	if stats.TotalSamples != 25 {
		t.Errorf("Expected TotalSamples 25, got %d", stats.TotalSamples)
	}

	if stats.ActualSamples != 25 {
		t.Errorf("Expected ActualSamples 25, got %d", stats.ActualSamples)
	}

	if stats.MaxSize != 100 {
		t.Errorf("Expected MaxSize 100, got %d", stats.MaxSize)
	}

	if stats.IsWrapped {
		t.Error("Expected IsWrapped false")
	}

	if stats.DBPath != dbPath {
		t.Errorf("Expected DBPath %s, got %s", dbPath, stats.DBPath)
	}
}

// TestCloseIdempotent tests that Close can be called multiple times
func TestCloseIdempotent(t *testing.T) {
	tmpDir := t.TempDir()
	dbPath := filepath.Join(tmpDir, "test.db")

	store, err := NewObservationStore(dbPath, 1000)
	if err != nil {
		t.Fatalf("Failed to create store: %v", err)
	}

	// Close multiple times
	err = store.Close()
	if err != nil {
		t.Errorf("First close failed: %v", err)
	}

	err = store.Close()
	if err != nil {
		t.Errorf("Second close failed: %v", err)
	}
}

// TestOperationsAfterClose tests that operations fail after close
func TestOperationsAfterClose(t *testing.T) {
	tmpDir := t.TempDir()
	dbPath := filepath.Join(tmpDir, "test.db")

	store, err := NewObservationStore(dbPath, 1000)
	if err != nil {
		t.Fatalf("Failed to create store: %v", err)
	}

	store.Close()

	// Try operations after close
	state := make([]float64, 64)

	err = store.StoreSample(state, 0)
	if err == nil {
		t.Error("Expected error for StoreSample after close")
	}

	_, err = store.GetBatch(10)
	if err == nil {
		t.Error("Expected error for GetBatch after close")
	}

	_, err = store.CountSamples()
	if err == nil {
		t.Error("Expected error for CountSamples after close")
	}
}

// BenchmarkStoreSample benchmarks sample storage
func BenchmarkStoreSample(b *testing.B) {
	tmpDir := b.TempDir()
	dbPath := filepath.Join(tmpDir, "bench.db")

	store, err := NewObservationStore(dbPath, 10000)
	if err != nil {
		b.Fatalf("Failed to create store: %v", err)
	}
	defer store.Close()

	state := make([]float64, 64)
	for i := range state {
		state[i] = rand.Float64()
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		err = store.StoreSample(state, i%4096)
		if err != nil {
			b.Fatalf("Failed to store sample: %v", err)
		}
	}
}

// BenchmarkGetBatch benchmarks batch retrieval
func BenchmarkGetBatch(b *testing.B) {
	tmpDir := b.TempDir()
	dbPath := filepath.Join(tmpDir, "bench.db")

	store, err := NewObservationStore(dbPath, 10000)
	if err != nil {
		b.Fatalf("Failed to create store: %v", err)
	}
	defer store.Close()

	// Pre-populate with samples
	for i := 0; i < 1000; i++ {
		state := make([]float64, 64)
		for j := range state {
			state[j] = rand.Float64()
		}
		err = store.StoreSample(state, i%4096)
		if err != nil {
			b.Fatalf("Failed to store sample: %v", err)
		}
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err = store.GetBatch(32)
		if err != nil {
			b.Fatalf("Failed to get batch: %v", err)
		}
	}
}

func init() {
	rand.Seed(time.Now().UnixNano())
}
