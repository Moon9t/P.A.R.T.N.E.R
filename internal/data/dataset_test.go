package data

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/notnil/chess"
)

func TestNewDataset(t *testing.T) {
	// Create temp directory
	tmpDir := t.TempDir()
	dbPath := filepath.Join(tmpDir, "test.db")

	ds, err := NewDataset(dbPath)
	if err != nil {
		t.Fatalf("Failed to create dataset: %v", err)
	}
	defer ds.Close()

	// Verify file was created
	if _, err := os.Stat(dbPath); os.IsNotExist(err) {
		t.Error("Database file was not created")
	}
}

func TestDatasetAdd(t *testing.T) {
	tmpDir := t.TempDir()
	dbPath := filepath.Join(tmpDir, "test.db")

	ds, err := NewDataset(dbPath)
	if err != nil {
		t.Fatalf("Failed to create dataset: %v", err)
	}
	defer ds.Close()

	// Create a sample entry
	entry := &DataEntry{
		StateTensor: make([]float32, NumChannels*BoardSize*BoardSize),
		FromSquare:  12,
		ToSquare:    28,
		GameID:      "test_game",
		MoveNumber:  1,
	}

	// Add entry
	if err := ds.Add(entry); err != nil {
		t.Fatalf("Failed to add entry: %v", err)
	}

	// Verify count
	count, err := ds.Count()
	if err != nil {
		t.Fatalf("Failed to get count: %v", err)
	}

	if count != 1 {
		t.Errorf("Expected count 1, got %d", count)
	}
}

func TestDatasetAddBatch(t *testing.T) {
	tmpDir := t.TempDir()
	dbPath := filepath.Join(tmpDir, "test.db")

	ds, err := NewDataset(dbPath)
	if err != nil {
		t.Fatalf("Failed to create dataset: %v", err)
	}
	defer ds.Close()

	// Create multiple entries
	entries := make([]*DataEntry, 10)
	for i := 0; i < 10; i++ {
		entries[i] = &DataEntry{
			StateTensor: make([]float32, NumChannels*BoardSize*BoardSize),
			FromSquare:  i,
			ToSquare:    i + 10,
			GameID:      "test_game",
			MoveNumber:  i,
		}
	}

	// Add batch
	if err := ds.AddBatch(entries); err != nil {
		t.Fatalf("Failed to add batch: %v", err)
	}

	// Verify count
	count, err := ds.Count()
	if err != nil {
		t.Fatalf("Failed to get count: %v", err)
	}

	if count != 10 {
		t.Errorf("Expected count 10, got %d", count)
	}
}

func TestLoadBatch(t *testing.T) {
	tmpDir := t.TempDir()
	dbPath := filepath.Join(tmpDir, "test.db")

	ds, err := NewDataset(dbPath)
	if err != nil {
		t.Fatalf("Failed to create dataset: %v", err)
	}
	defer ds.Close()

	// Add test data
	totalEntries := 20
	entries := make([]*DataEntry, totalEntries)
	for i := 0; i < totalEntries; i++ {
		entries[i] = &DataEntry{
			StateTensor: make([]float32, NumChannels*BoardSize*BoardSize),
			FromSquare:  i,
			ToSquare:    i + 10,
			GameID:      "test_game",
			MoveNumber:  i,
		}
	}

	if err := ds.AddBatch(entries); err != nil {
		t.Fatalf("Failed to add batch: %v", err)
	}

	// Test loading first batch
	batch1, err := ds.LoadBatch(0, 10)
	if err != nil {
		t.Fatalf("Failed to load batch: %v", err)
	}

	if len(batch1) != 10 {
		t.Errorf("Expected batch size 10, got %d", len(batch1))
	}

	// Test loading second batch
	batch2, err := ds.LoadBatch(10, 10)
	if err != nil {
		t.Fatalf("Failed to load second batch: %v", err)
	}

	if len(batch2) != 10 {
		t.Errorf("Expected batch size 10, got %d", len(batch2))
	}

	// Verify data
	if batch1[0].FromSquare != 0 {
		t.Errorf("Expected first entry FromSquare=0, got %d", batch1[0].FromSquare)
	}

	if batch2[0].FromSquare != 10 {
		t.Errorf("Expected second batch first entry FromSquare=10, got %d", batch2[0].FromSquare)
	}
}

func TestVerifyIntegrity(t *testing.T) {
	tmpDir := t.TempDir()
	dbPath := filepath.Join(tmpDir, "test.db")

	ds, err := NewDataset(dbPath)
	if err != nil {
		t.Fatalf("Failed to create dataset: %v", err)
	}
	defer ds.Close()

	// Add valid entries
	game := chess.NewGame()
	if err := game.MoveStr("e4"); err != nil {
		t.Fatalf("Failed to make move: %v", err)
	}

	board := game.Position().Board()
	tensor, err := TensorizeBoard(board)
	if err != nil {
		t.Fatalf("Failed to tensorize board: %v", err)
	}

	entry := &DataEntry{
		StateTensor: TensorToFlatArray(tensor),
		FromSquare:  12,
		ToSquare:    28,
		GameID:      "test_game",
		MoveNumber:  1,
	}

	if err := ds.Add(entry); err != nil {
		t.Fatalf("Failed to add entry: %v", err)
	}

	// Verify integrity
	if err := ds.VerifyIntegrity(); err != nil {
		t.Errorf("Integrity check failed: %v", err)
	}
}

func TestVerifyIntegrity_InvalidData(t *testing.T) {
	tmpDir := t.TempDir()
	dbPath := filepath.Join(tmpDir, "test.db")

	ds, err := NewDataset(dbPath)
	if err != nil {
		t.Fatalf("Failed to create dataset: %v", err)
	}
	defer ds.Close()

	// Add entry with invalid tensor size
	entry := &DataEntry{
		StateTensor: make([]float32, 100), // Wrong size!
		FromSquare:  12,
		ToSquare:    28,
		GameID:      "test_game",
		MoveNumber:  1,
	}

	if err := ds.Add(entry); err != nil {
		t.Fatalf("Failed to add entry: %v", err)
	}

	// Verify integrity should fail
	if err := ds.VerifyIntegrity(); err == nil {
		t.Error("Expected integrity check to fail for invalid tensor size")
	}
}

func TestGetStats(t *testing.T) {
	tmpDir := t.TempDir()
	dbPath := filepath.Join(tmpDir, "test.db")

	ds, err := NewDataset(dbPath)
	if err != nil {
		t.Fatalf("Failed to create dataset: %v", err)
	}
	defer ds.Close()

	// Add some entries
	entries := make([]*DataEntry, 5)
	for i := 0; i < 5; i++ {
		entries[i] = &DataEntry{
			StateTensor: make([]float32, NumChannels*BoardSize*BoardSize),
			FromSquare:  i,
			ToSquare:    i + 10,
		}
	}

	if err := ds.AddBatch(entries); err != nil {
		t.Fatalf("Failed to add batch: %v", err)
	}

	// Get stats
	stats, err := ds.GetStats()
	if err != nil {
		t.Fatalf("Failed to get stats: %v", err)
	}

	if stats.TotalEntries != 5 {
		t.Errorf("Expected 5 entries, got %d", stats.TotalEntries)
	}

	if stats.FilePath != dbPath {
		t.Errorf("Expected path %s, got %s", dbPath, stats.FilePath)
	}

	if stats.FileSize <= 0 {
		t.Error("Expected positive file size")
	}
}

func TestClear(t *testing.T) {
	tmpDir := t.TempDir()
	dbPath := filepath.Join(tmpDir, "test.db")

	ds, err := NewDataset(dbPath)
	if err != nil {
		t.Fatalf("Failed to create dataset: %v", err)
	}
	defer ds.Close()

	// Add entries
	entry := &DataEntry{
		StateTensor: make([]float32, NumChannels*BoardSize*BoardSize),
		FromSquare:  12,
		ToSquare:    28,
	}

	if err := ds.Add(entry); err != nil {
		t.Fatalf("Failed to add entry: %v", err)
	}

	// Verify count
	count, _ := ds.Count()
	if count != 1 {
		t.Errorf("Expected count 1 before clear, got %d", count)
	}

	// Clear
	if err := ds.Clear(); err != nil {
		t.Fatalf("Failed to clear dataset: %v", err)
	}

	// Verify count after clear
	count, _ = ds.Count()
	if count != 0 {
		t.Errorf("Expected count 0 after clear, got %d", count)
	}
}

func TestLoadBatchBeyondEnd(t *testing.T) {
	tmpDir := t.TempDir()
	dbPath := filepath.Join(tmpDir, "test.db")

	ds, err := NewDataset(dbPath)
	if err != nil {
		t.Fatalf("Failed to create dataset: %v", err)
	}
	defer ds.Close()

	// Add 5 entries
	entries := make([]*DataEntry, 5)
	for i := 0; i < 5; i++ {
		entries[i] = &DataEntry{
			StateTensor: make([]float32, NumChannels*BoardSize*BoardSize),
			FromSquare:  i,
			ToSquare:    i + 10,
		}
	}

	if err := ds.AddBatch(entries); err != nil {
		t.Fatalf("Failed to add batch: %v", err)
	}

	// Try to load beyond end
	batch, err := ds.LoadBatch(3, 10) // Asks for 10 but only 2 available
	if err != nil {
		t.Fatalf("Failed to load batch: %v", err)
	}

	if len(batch) != 2 {
		t.Errorf("Expected 2 entries (all remaining), got %d", len(batch))
	}
}
