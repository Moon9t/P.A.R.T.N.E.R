

# Chess Dataset Ingestion Package

A complete Go package for ingesting chess games from PGN files and converting them into training data for neural networks.

## Features

- **PGN Parsing**: Parse `.pgn` files using `github.com/notnil/chess`
- **Board Tensorization**: Convert chess positions to `[12][8][8]` float32 tensors
- **Move Encoding**: Encode moves as (from_square, to_square) integer pairs (0-63)
- **Efficient Storage**: Store data in BoltDB for on-disk persistence
- **Batch Loading**: Stream data in batches to avoid memory overload
- **Integrity Verification**: Validate dataset consistency
- **Parallel Processing**: Multi-threaded ingestion with configurable worker pools

## Package Structure

```
internal/data/
├── pgn_parser.go      # PGN file parsing
├── tensorize.go       # Board-to-tensor conversion
├── dataset.go         # BoltDB storage management
├── ingestion.go       # Complete ingestion pipeline
├── pgn_parser_test.go # PGN parsing tests
├── tensorize_test.go  # Tensorization tests
└── dataset_test.go    # Dataset storage tests
```

## Tensor Format

Each chess position is encoded as a `[12][8][8]` tensor:

- **Channels 0-5**: White pieces (Pawn, Knight, Bishop, Rook, Queen, King)
- **Channels 6-11**: Black pieces (Pawn, Knight, Bishop, Rook, Queen, King)
- **Values**: Binary (0.0 or 1.0) indicating piece presence

### Example

```go
tensor := TensorizeBoard(board)
// tensor[0][6][4] == 1.0  means white pawn at e2
// tensor[6][1][4] == 1.0  means black pawn at e7
```

## Move Encoding

Moves are encoded as pairs of square indices:

- **From Square**: 0-63 (a1=0, b1=1, ..., h8=63)
- **To Square**: 0-63

```go
fromSquare, toSquare, err := EncodeMoveLabel(move)
// e2e4 -> (12, 28)
```

## Usage

### 1. Ingest PGN Files

```go
config := &data.IngestionConfig{
    PGNPath:        "games.pgn",
    DatasetPath:    "chess_dataset.db",
    MaxGames:       1000,
    MaxPositions:   10000,
    SkipInvalid:    true,
    BatchSize:      100,
    WorkerPoolSize: 4,
    Verbose:        true,
}

ingestor, err := data.NewIngestor(config)
if err != nil {
    log.Fatal(err)
}
defer ingestor.Close()

stats, err := ingestor.Ingest()
if err != nil {
    log.Fatal(err)
}

fmt.Printf("Ingested %d positions from %d games\n", 
    stats.PositionsIngested, stats.GamesProcessed)
```

### 2. Load Training Data

```go
dataset, err := data.NewDataset("chess_dataset.db")
if err != nil {
    log.Fatal(err)
}
defer dataset.Close()

// Load batch of 100 positions
entries, err := dataset.LoadBatch(0, 100)
if err != nil {
    log.Fatal(err)
}

for _, entry := range entries {
    // Convert flat array back to tensor
    tensor, err := data.FlatArrayToTensor(entry.StateTensor)
    if err != nil {
        log.Fatal(err)
    }
    
    // Use tensor and move labels for training
    trainModel(tensor, entry.FromSquare, entry.ToSquare)
}
```

### 3. Verify Dataset Integrity

```go
dataset, err := data.NewDataset("chess_dataset.db")
if err != nil {
    log.Fatal(err)
}
defer dataset.Close()

if err := dataset.VerifyIntegrity(); err != nil {
    log.Fatal("Dataset integrity check failed:", err)
}

fmt.Println("Dataset integrity verified ✓")
```

### 4. Get Dataset Statistics

```go
stats, err := dataset.GetStats()
if err != nil {
    log.Fatal(err)
}

fmt.Printf("Total entries: %d\n", stats.TotalEntries)
fmt.Printf("File size: %.2f MB\n", float64(stats.FileSize)/1024/1024)
```

## Command-Line Tools

### Ingest PGN Files

```bash
# Build the ingestion tool
go build -o bin/ingest-pgn cmd/ingest-pgn/main.go

# Ingest a PGN file
./bin/ingest-pgn -pgn=data/sample_games.pgn -dataset=data/chess_dataset.db

# With limits
./bin/ingest-pgn -pgn=large_file.pgn -max-games=1000 -max-positions=50000

# Show dataset statistics
./bin/ingest-pgn -dataset=data/chess_dataset.db -stats

# Verify integrity after ingestion
./bin/ingest-pgn -pgn=games.pgn -dataset=output.db -verify
```

### Load and Display Dataset

```bash
# Build main program
go build -o bin/partner cmd/partner/main.go

# Load and display first 5 positions
./bin/partner -mode=load-dataset -dataset=data/chess_dataset.db
```

## API Reference

### PGN Parsing

```go
// Parse PGN file
parser := data.NewPGNParser("games.pgn")
games, err := parser.ParsePGN()

// Extract positions from a game
positions, err := data.ExtractPositions(game)
```

### Tensorization

```go
// Convert board to tensor
tensor, err := data.TensorizeBoard(board)

// Encode move
fromSquare, toSquare, err := data.EncodeMoveLabel(move)

// Decode move
from, to, err := data.DecodeMoveLabel(12, 28)

// Validate tensor
err := data.ValidateTensor(tensor)

// Flatten tensor for storage
flat := data.TensorToFlatArray(tensor)

// Reconstruct tensor
tensor, err := data.FlatArrayToTensor(flat)
```

### Dataset Management

```go
// Create or open dataset
dataset, err := data.NewDataset("chess.db")

// Add single entry
entry := &data.DataEntry{
    StateTensor: flat,
    FromSquare:  12,
    ToSquare:    28,
    GameID:      "game_1",
    MoveNumber:  1,
}
err := dataset.Add(entry)

// Add batch
err := dataset.AddBatch(entries)

// Load batch (streaming)
entries, err := dataset.LoadBatch(offset, batchSize)

// Get count
count, err := dataset.Count()

// Verify integrity
err := dataset.VerifyIntegrity()

// Get statistics
stats, err := dataset.GetStats()

// Clear dataset
err := dataset.Clear()
```

## Testing

```bash
# Run all tests
go test ./internal/data/...

# Run with verbose output
go test -v ./internal/data/...

# Run specific test
go test -v ./internal/data/ -run TestTensorizeBoard

# Check coverage
go test -cover ./internal/data/...
```

## Performance Characteristics

- **Memory Usage**: Streaming architecture supports datasets larger than RAM
- **Ingestion Speed**: ~1000-5000 positions/second (depends on CPU and disk)
- **Parallel Processing**: Configurable worker pools for multi-core systems
- **Batch Size**: Recommended 100-1000 for optimal disk I/O
- **Dataset Size**: Linear growth (~1.5KB per position)

## CPU Optimization

This package is designed for CPU-only systems:

- No CUDA dependencies
- Efficient batch loading
- Parallel ingestion with worker pools
- Memory-mapped database access via BoltDB

## Example Output

```
Initializing dataset ingestion...
  PGN file: data/sample_games.pgn
  Dataset: data/chess_dataset.db
  Workers: 4

Starting ingestion...
Parsed 3 games from sample_games.pgn

Ingestion complete:
  Games processed: 3/3
  Positions ingested: 65
  Positions skipped: 0

============================================================
Ingestion Complete
============================================================
Games processed:     3 / 3
Positions ingested:  65
Positions skipped:   0

Dataset ready for training!
```

## Dependencies

- `github.com/notnil/chess` - Chess game logic and PGN parsing
- `go.etcd.io/bbolt` - Embedded key-value database

## Error Handling

The package follows Go best practices:

- All functions return explicit errors
- Invalid data can be skipped with `SkipInvalid` flag
- Integrity checks validate dataset consistency
- Detailed error messages for debugging

## License

Part of the P.A.R.T.N.E.R project - see main LICENSE file.
