# Chess Dataset Quick Reference

## Build Commands

```bash
# Build ingestion tool
go build -o bin/ingest-pgn cmd/ingest-pgn/main.go

# Build main application
go build -o bin/partner cmd/partner/main.go
```

## Common Commands

### Ingest PGN File
```bash
./bin/ingest-pgn -pgn=data/sample_games.pgn -dataset=data/chess_dataset.db
```

### View Statistics
```bash
./bin/ingest-pgn -dataset=data/chess_dataset.db -stats
```

### Verify Integrity
```bash
./bin/ingest-pgn -pgn=games.pgn -dataset=output.db -verify
```

### Load and Display
```bash
ASSUME_NO_MOVING_GC_UNSAFE_RISK_IT_WITH=go1.25 \
  ./bin/partner -mode=load-dataset -dataset=data/chess_dataset.db
```

## Tensor Format

- **Shape:** `[12][8][8]` float32
- **Size:** 768 values per position
- **Channels:** 0-5 White pieces, 6-11 Black pieces
- **Values:** 0.0 or 1.0 (one-hot encoding)

## Move Encoding

- **From/To:** Integer 0-63
- **Mapping:** a1=0, b1=1, ..., h1=7, a2=8, ..., h8=63

## API Quick Reference

```go
// Parse PGN
parser := data.NewPGNParser("games.pgn")
games, err := parser.ParsePGN()

// Tensorize board
tensor, err := data.TensorizeBoard(board)

// Create dataset
ds, err := data.NewDataset("chess.db")
defer ds.Close()

// Load batch
entries, err := ds.LoadBatch(0, 100)
```

## Test Command

```bash
go test -v -cover ./internal/data/...
```

## Project Structure

```
internal/data/
├── pgn_parser.go      # PGN parsing
├── tensorize.go       # Board encoding
├── dataset.go         # BoltDB storage
├── ingestion.go       # Full pipeline
├── *_test.go          # Unit tests
└── README.md          # Full documentation
```
