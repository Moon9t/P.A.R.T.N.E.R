package training

import (
	"encoding/json"
	"os"
	"testing"
	"time"
	
	"github.com/thyrook/partner/internal/model"
)

// Helper function to create test moves
func makeTestMove(notation string) Move {
	index, _ := model.EncodeMove(notation)
	return Move{
		Index:      index,
		Notation:   notation,
		FromSquare: notation[0:2],
		ToSquare:   notation[2:4],
		Confidence: 0.5,
	}
}

func TestReplayBuffer(t *testing.T) {
	buffer := NewReplayBuffer(10)

	// Test adding entries
	t.Run("AddEntries", func(t *testing.T) {
		testMoves := []string{"e2e4", "e7e5", "g1f3", "b8c6", "f1c4"}
		for _, notation := range testMoves {
			move := makeTestMove(notation)
			entry := ReplayEntry{
				PredictedMove: move,
				ActualMove:    move,
			}
			buffer.Add(entry)
		}

		if len(buffer.Entries) != 5 {
			t.Errorf("Expected 5 entries, got %d", len(buffer.Entries))
		}

		if buffer.CorrectPredictions != 5 {
			t.Errorf("Expected 5 correct predictions, got %d", buffer.CorrectPredictions)
		}
	})

	// Test incorrect predictions
	t.Run("IncorrectPredictions", func(t *testing.T) {
		entry := ReplayEntry{
			PredictedMove: makeTestMove("e2e4"),
			ActualMove:    makeTestMove("d2d4"),
		}
		buffer.Add(entry)

		stats := buffer.GetStats()
		if stats.Accuracy >= 1.0 {
			t.Error("Accuracy should be less than 100% after incorrect prediction")
		}
	})

	// Test buffer overflow
	t.Run("BufferOverflow", func(t *testing.T) {
		testMoves := []string{
			"a2a3", "a2a4", "b2b3", "b2b4", "c2c3", "c2c4", "d2d3", "d2d4",
			"e2e3", "e2e4", "f2f3", "f2f4", "g2g3", "g2g4", "h2h3", "h2h4",
			"a7a6", "a7a5", "b7b6", "b7b5",
		}
		for _, notation := range testMoves {
			move := makeTestMove(notation)
			entry := ReplayEntry{
				PredictedMove: move,
				ActualMove:    move,
			}
			buffer.Add(entry)
		}

		if len(buffer.Entries) > 10 {
			t.Errorf("Buffer exceeded max size: %d", len(buffer.Entries))
		}
	})
}

func TestReplayStats(t *testing.T) {
	buffer := NewReplayBuffer(100)

	// Add mix of correct and incorrect
	correctMoves := []string{"e2e4", "e7e5", "g1f3", "b8c6", "f1c4"}
	incorrectPairs := []struct{ pred, actual string }{
		{"e2e4", "d2d4"},
		{"e7e5", "e7e6"},
		{"g1f3", "g1e2"},
		{"b8c6", "b8d7"},
		{"f1c4", "f1b5"},
	}

	// Add 25 correct predictions
	for i := 0; i < 5; i++ {
		for _, notation := range correctMoves {
			move := makeTestMove(notation)
			entry := ReplayEntry{
				PredictedMove: move,
				ActualMove:    move,
			}
			buffer.Add(entry)
		}
	}

	// Add 25 incorrect predictions
	for i := 0; i < 5; i++ {
		for _, pair := range incorrectPairs {
			entry := ReplayEntry{
				PredictedMove: makeTestMove(pair.pred),
				ActualMove:    makeTestMove(pair.actual),
			}
			buffer.Add(entry)
		}
	}

	stats := buffer.GetStats()

	t.Run("Accuracy", func(t *testing.T) {
		expected := 0.5
		if stats.Accuracy != expected {
			t.Errorf("Expected accuracy %.2f, got %.2f", expected, stats.Accuracy)
		}
	})

	t.Run("TotalEntries", func(t *testing.T) {
		if stats.TotalEntries != 50 {
			t.Errorf("Expected 50 entries, got %d", stats.TotalEntries)
		}
	})

	t.Run("AverageReward", func(t *testing.T) {
		// Should be 0 (half +1, half -1)
		if stats.AverageReward != 0.0 {
			t.Errorf("Expected average reward 0.0, got %.2f", stats.AverageReward)
		}
	})
}

func TestRewardWeightedSample(t *testing.T) {
	buffer := NewReplayBuffer(100)

	// Add correct entries
	correctMoves := []string{"e2e4", "e7e5", "g1f3", "b8c6", "f1c4"}
	for i := 0; i < 4; i++ {
		for _, notation := range correctMoves {
			move := makeTestMove(notation)
			entry := ReplayEntry{
				PredictedMove: move,
				ActualMove:    move,
			}
			buffer.Add(entry)
		}
	}

	// Add incorrect entries
	incorrectPairs := []struct{ pred, actual string }{
		{"e2e4", "d2d4"},
		{"e7e5", "e7e6"},
	}
	for i := 0; i < 5; i++ {
		for _, pair := range incorrectPairs {
			entry := ReplayEntry{
				PredictedMove: makeTestMove(pair.pred),
				ActualMove:    makeTestMove(pair.actual),
			}
			buffer.Add(entry)
		}
	}

	sample := buffer.GetRewardWeightedSample(10)

	if len(sample) == 0 {
		t.Error("Expected non-empty sample")
	}

	if len(sample) > 10 {
		t.Errorf("Sample size exceeded requested: %d", len(sample))
	}
}

func TestBalancedSample(t *testing.T) {
	buffer := NewReplayBuffer(100)

	// Add more correct than incorrect
	correctMoves := []string{"e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "f8c5"}
	for i := 0; i < 5; i++ {
		for _, notation := range correctMoves {
			move := makeTestMove(notation)
			entry := ReplayEntry{
				PredictedMove: move,
				ActualMove:    move,
			}
			buffer.Add(entry)
		}
	}

	// Add fewer incorrect entries
	incorrectPairs := []struct{ pred, actual string }{
		{"e2e4", "d2d4"},
		{"e7e5", "e7e6"},
	}
	for i := 0; i < 5; i++ {
		for _, pair := range incorrectPairs {
			entry := ReplayEntry{
				PredictedMove: makeTestMove(pair.pred),
				ActualMove:    makeTestMove(pair.actual),
			}
			buffer.Add(entry)
		}
	}

	sample := buffer.GetBalancedSample(20)

	correctCount := 0
	incorrectCount := 0
	for _, entry := range sample {
		if entry.IsCorrect {
			correctCount++
		} else {
			incorrectCount++
		}
	}

	// Should be roughly balanced
	diff := correctCount - incorrectCount
	if diff < -5 || diff > 5 {
		t.Errorf("Sample not balanced: %d correct, %d incorrect", correctCount, incorrectCount)
	}
}

func TestReplayStorage(t *testing.T) {
	// Create temporary storage
	tmpDir := t.TempDir()
	dbPath := tmpDir + "/test_replay.db"
	jsonlDir := tmpDir + "/jsonl"

	storage, err := NewReplayStorage(dbPath, jsonlDir)
	if err != nil {
		t.Fatalf("Failed to create storage: %v", err)
	}
	defer storage.Close()

	t.Run("Store", func(t *testing.T) {
		move := makeTestMove("e2e4")
		entry := ReplayEntry{
			PredictedMove: move,
			ActualMove:    move,
			Timestamp:     time.Now().Unix(),
		}

		if err := storage.Store(entry); err != nil {
			t.Errorf("Failed to store entry: %v", err)
		}
	})

	t.Run("StoreBatch", func(t *testing.T) {
		testMoves := []string{"e2e4", "e7e5", "g1f3", "b8c6", "f1c4"}
		entries := make([]ReplayEntry, len(testMoves))
		for i, notation := range testMoves {
			move := makeTestMove(notation)
			entries[i] = ReplayEntry{
				PredictedMove: move,
				ActualMove:    move,
				Timestamp:     time.Now().Unix(),
			}
		}

		if err := storage.StoreBatch(entries); err != nil {
			t.Errorf("Failed to store batch: %v", err)
		}
	})

	t.Run("LoadAll", func(t *testing.T) {
		entries, err := storage.LoadAll()
		if err != nil {
			t.Errorf("Failed to load entries: %v", err)
		}

		if len(entries) < 6 {
			t.Errorf("Expected at least 6 entries, got %d", len(entries))
		}
	})

	t.Run("Count", func(t *testing.T) {
		count, err := storage.Count()
		if err != nil {
			t.Errorf("Failed to count entries: %v", err)
		}

		if count < 6 {
			t.Errorf("Expected at least 6 entries, got %d", count)
		}
	})

	t.Run("ExportJSONL", func(t *testing.T) {
		if err := storage.ExportToJSONL("test_export.jsonl"); err != nil {
			t.Errorf("Failed to export to JSONL: %v", err)
		}

		// Check file exists
		filepath := jsonlDir + "/test_export.jsonl"
		if _, err := os.Stat(filepath); os.IsNotExist(err) {
			t.Error("JSONL file was not created")
		}
	})

	t.Run("Metadata", func(t *testing.T) {
		key := "test_key"
		value := "test_value"

		if err := storage.SetMetadata(key, value); err != nil {
			t.Errorf("Failed to set metadata: %v", err)
		}

		retrieved, err := storage.GetMetadata(key)
		if err != nil {
			t.Errorf("Failed to get metadata: %v", err)
		}

		if retrieved != value {
			t.Errorf("Expected %s, got %s", value, retrieved)
		}
	})
}

func TestReplayBufferJSON(t *testing.T) {
	buffer := NewReplayBuffer(10)

	testMoves := []string{"e2e4", "e7e5", "g1f3", "b8c6", "f1c4"}
	for _, notation := range testMoves {
		move := makeTestMove(notation)
		entry := ReplayEntry{
			PredictedMove: move,
			ActualMove:    move,
		}
		buffer.Add(entry)
	}

	t.Run("ToJSON", func(t *testing.T) {
		data, err := buffer.ToJSON()
		if err != nil {
			t.Errorf("Failed to serialize buffer: %v", err)
		}

		if len(data) == 0 {
			t.Error("Serialized data is empty")
		}
	})

	t.Run("FromJSON", func(t *testing.T) {
		data, _ := buffer.ToJSON()

		newBuffer := NewReplayBuffer(10)
		if err := newBuffer.FromJSON(data); err != nil {
			t.Errorf("Failed to deserialize buffer: %v", err)
		}

		if len(newBuffer.Entries) != len(buffer.Entries) {
			t.Errorf("Entry count mismatch: expected %d, got %d",
				len(buffer.Entries), len(newBuffer.Entries))
		}
	})
}

func TestReplayEntryReward(t *testing.T) {
	buffer := NewReplayBuffer(10)

	t.Run("CorrectPrediction", func(t *testing.T) {
		move := makeTestMove("e2e4")
		entry := ReplayEntry{
			PredictedMove: move,
			ActualMove:    move,
		}
		buffer.Add(entry)

		lastEntry := buffer.Entries[len(buffer.Entries)-1]
		if lastEntry.Reward != 1.0 {
			t.Errorf("Expected reward 1.0, got %.2f", lastEntry.Reward)
		}
		if !lastEntry.IsCorrect {
			t.Error("Expected IsCorrect to be true")
		}
	})

	t.Run("IncorrectPrediction", func(t *testing.T) {
		entry := ReplayEntry{
			PredictedMove: makeTestMove("e2e4"),
			ActualMove:    makeTestMove("d2d4"),
		}
		buffer.Add(entry)

		lastEntry := buffer.Entries[len(buffer.Entries)-1]
		if lastEntry.Reward != -1.0 {
			t.Errorf("Expected reward -1.0, got %.2f", lastEntry.Reward)
		}
		if lastEntry.IsCorrect {
			t.Error("Expected IsCorrect to be false")
		}
	})
}

func TestMoveEncoding(t *testing.T) {
	t.Run("MoveNotation", func(t *testing.T) {
		move := makeTestMove("e2e4")

		data, err := json.Marshal(move)
		if err != nil {
			t.Errorf("Failed to marshal move: %v", err)
		}

		var decoded Move
		if err := json.Unmarshal(data, &decoded); err != nil {
			t.Errorf("Failed to unmarshal move: %v", err)
		}

		if decoded.Notation != move.Notation {
			t.Error("Move encoding/decoding mismatch")
		}
	})
}
