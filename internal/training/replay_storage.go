package training

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"time"

	bolt "go.etcd.io/bbolt"
)

const (
	replayBucketName = "replays"
	metaBucketName   = "metadata"
)

// ReplayStorage handles persistent storage of replay entries
type ReplayStorage struct {
	db       *bolt.DB
	jsonlDir string // For JSONL fallback/export
}

// NewReplayStorage creates a new replay storage instance
func NewReplayStorage(dbPath, jsonlDir string) (*ReplayStorage, error) {
	// Ensure directories exist
	if err := os.MkdirAll(filepath.Dir(dbPath), 0755); err != nil {
		return nil, fmt.Errorf("failed to create db directory: %w", err)
	}

	if err := os.MkdirAll(jsonlDir, 0755); err != nil {
		return nil, fmt.Errorf("failed to create jsonl directory: %w", err)
	}

	// Open BoltDB
	db, err := bolt.Open(dbPath, 0600, &bolt.Options{Timeout: 1 * time.Second})
	if err != nil {
		return nil, fmt.Errorf("failed to open database: %w", err)
	}

	// Create buckets
	err = db.Update(func(tx *bolt.Tx) error {
		if _, err := tx.CreateBucketIfNotExists([]byte(replayBucketName)); err != nil {
			return err
		}
		if _, err := tx.CreateBucketIfNotExists([]byte(metaBucketName)); err != nil {
			return err
		}
		return nil
	})

	if err != nil {
		db.Close()
		return nil, fmt.Errorf("failed to create buckets: %w", err)
	}

	return &ReplayStorage{
		db:       db,
		jsonlDir: jsonlDir,
	}, nil
}

// Store saves a replay entry to persistent storage
func (rs *ReplayStorage) Store(entry ReplayEntry) error {
	// Generate key (timestamp + random suffix for uniqueness)
	key := []byte(fmt.Sprintf("%d_%d", entry.Timestamp, time.Now().UnixNano()))

	// Serialize entry
	data, err := json.Marshal(entry)
	if err != nil {
		return fmt.Errorf("failed to marshal entry: %w", err)
	}

	// Store in BoltDB
	err = rs.db.Update(func(tx *bolt.Tx) error {
		bucket := tx.Bucket([]byte(replayBucketName))
		return bucket.Put(key, data)
	})

	if err != nil {
		return fmt.Errorf("failed to store entry: %w", err)
	}

	return nil
}

// StoreBatch stores multiple entries efficiently
func (rs *ReplayStorage) StoreBatch(entries []ReplayEntry) error {
	return rs.db.Update(func(tx *bolt.Tx) error {
		bucket := tx.Bucket([]byte(replayBucketName))

		for _, entry := range entries {
			key := []byte(fmt.Sprintf("%d_%d", entry.Timestamp, time.Now().UnixNano()))
			data, err := json.Marshal(entry)
			if err != nil {
				return fmt.Errorf("failed to marshal entry: %w", err)
			}

			if err := bucket.Put(key, data); err != nil {
				return err
			}
		}

		return nil
	})
}

// LoadAll loads all replay entries from storage
func (rs *ReplayStorage) LoadAll() ([]ReplayEntry, error) {
	var entries []ReplayEntry

	err := rs.db.View(func(tx *bolt.Tx) error {
		bucket := tx.Bucket([]byte(replayBucketName))

		return bucket.ForEach(func(k, v []byte) error {
			var entry ReplayEntry
			if err := json.Unmarshal(v, &entry); err != nil {
				return err
			}
			entries = append(entries, entry)
			return nil
		})
	})

	if err != nil {
		return nil, fmt.Errorf("failed to load entries: %w", err)
	}

	return entries, nil
}

// LoadRecent loads the N most recent entries
func (rs *ReplayStorage) LoadRecent(n int) ([]ReplayEntry, error) {
	entries, err := rs.LoadAll()
	if err != nil {
		return nil, err
	}

	// Sort by timestamp (descending) and take first N
	// For now, return last N entries
	if len(entries) <= n {
		return entries, nil
	}

	return entries[len(entries)-n:], nil
}

// Count returns the number of stored entries
func (rs *ReplayStorage) Count() (int, error) {
	var count int

	err := rs.db.View(func(tx *bolt.Tx) error {
		bucket := tx.Bucket([]byte(replayBucketName))
		stats := bucket.Stats()
		count = stats.KeyN
		return nil
	})

	return count, err
}

// ExportToJSONL exports entries to a JSONL file
func (rs *ReplayStorage) ExportToJSONL(filename string) error {
	entries, err := rs.LoadAll()
	if err != nil {
		return err
	}

	filepath := filepath.Join(rs.jsonlDir, filename)
	file, err := os.Create(filepath)
	if err != nil {
		return fmt.Errorf("failed to create file: %w", err)
	}
	defer file.Close()

	encoder := json.NewEncoder(file)
	for _, entry := range entries {
		if err := encoder.Encode(entry); err != nil {
			return fmt.Errorf("failed to encode entry: %w", err)
		}
	}

	return nil
}

// ImportFromJSONL imports entries from a JSONL file
func (rs *ReplayStorage) ImportFromJSONL(filename string) error {
	filepath := filepath.Join(rs.jsonlDir, filename)
	file, err := os.Open(filepath)
	if err != nil {
		return fmt.Errorf("failed to open file: %w", err)
	}
	defer file.Close()

	decoder := json.NewDecoder(file)
	var entries []ReplayEntry

	for decoder.More() {
		var entry ReplayEntry
		if err := decoder.Decode(&entry); err != nil {
			return fmt.Errorf("failed to decode entry: %w", err)
		}
		entries = append(entries, entry)
	}

	return rs.StoreBatch(entries)
}

// Clear removes all entries from storage
func (rs *ReplayStorage) Clear() error {
	return rs.db.Update(func(tx *bolt.Tx) error {
		if err := tx.DeleteBucket([]byte(replayBucketName)); err != nil {
			return err
		}
		_, err := tx.CreateBucket([]byte(replayBucketName))
		return err
	})
}

// GetMetadata retrieves metadata value
func (rs *ReplayStorage) GetMetadata(key string) (string, error) {
	var value string

	err := rs.db.View(func(tx *bolt.Tx) error {
		bucket := tx.Bucket([]byte(metaBucketName))
		data := bucket.Get([]byte(key))
		if data == nil {
			return fmt.Errorf("key not found: %s", key)
		}
		value = string(data)
		return nil
	})

	return value, err
}

// SetMetadata stores metadata value
func (rs *ReplayStorage) SetMetadata(key, value string) error {
	return rs.db.Update(func(tx *bolt.Tx) error {
		bucket := tx.Bucket([]byte(metaBucketName))
		return bucket.Put([]byte(key), []byte(value))
	})
}

// Close closes the storage
func (rs *ReplayStorage) Close() error {
	return rs.db.Close()
}

// Backup creates a backup of the database
func (rs *ReplayStorage) Backup(backupPath string) error {
	return rs.db.View(func(tx *bolt.Tx) error {
		return tx.CopyFile(backupPath, 0600)
	})
}
