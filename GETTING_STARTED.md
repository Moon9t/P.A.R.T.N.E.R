# Getting Started with P.A.R.T.N.E.R

Complete guide to get up and running with P.A.R.T.N.E.R in minutes.

## Prerequisites

### Required
- **Go 1.21+** - [Download](https://go.dev/doc/install)
- **OpenCV 4.x** - Computer vision library
- **8GB RAM** - Minimum for training
- **Linux/macOS** - Primary platforms

### Install OpenCV

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install -y libopencv-dev libopencv-contrib-dev pkg-config
```

**macOS:**
```bash
brew install opencv
```

## Quick Start (60 seconds)

### 1. Check System Status
```bash
make status
```

This verifies all dependencies and shows what's ready.

### 2. Run the Demo
```bash
make demo
```

Runs a complete pipeline with sample data in under 60 seconds:
- Creates test PGN games
- Ingests into database
- Trains a CNN model
- Runs self-improvement
- Shows results

### 3. Interactive Guide
```bash
make quick-start
```

Interactive menu with options:
1. Complete workflow (PGN → Train → Self-Improve)
2. Just train a model
3. Run self-improvement
4. Live chess analysis
5. Test existing model
6. Build tools only

## Complete Workflow

### Step 1: Obtain Chess Games

Get PGN files from:
- [Lichess Database](https://database.lichess.org/)
- [FICS Games Database](https://www.ficsgames.org/)
- Your own games from Chess.com or Lichess

Example:
```bash
wget https://database.lichess.org/standard/lichess_db_standard_rated_2024-01.pgn.zst
unzstd lichess_db_standard_rated_2024-01.pgn.zst
```

### Step 2: Ingest PGN Data

```bash
./run.sh ingest-pgn --input games.pgn --output data/positions.db
```

This extracts positions and moves into a BoltDB database.

### Step 3: Train CNN Model

```bash
./run.sh train-cnn \
  --dataset data/positions.db \
  --model data/models/chess_cnn.bin \
  --epochs 50 \
  --batch-size 32
```

**Training features:**
- Data augmentation (flips, color inversion)
- Learning rate scheduling
- Real-time loss/accuracy tracking
- Automatic checkpointing

### Step 4: Self-Improvement

```bash
./run.sh self-improvement \
  --model data/models/chess_cnn.bin \
  --dataset data/positions.db \
  --observations 100
```

The system:
1. Loads the trained model
2. Makes predictions on real positions
3. Compares to actual moves
4. Stores observations in replay buffer
5. Automatically trains when buffer is full

### Step 5: Live Analysis

```bash
make run-live-chess
```

Or with custom settings:
```bash
./run.sh live-chess \
  --model data/models/chess_cnn.bin \
  --x 100 --y 100 \
  --width 800 --height 800 \
  --fps 2 --top 5
```

**Live analysis provides:**
- Real-time board capture
- Position detection
- Top-K move predictions
- Confidence visualization

## Tools Overview

All tools can be run via `./run.sh <tool-name> [args]`

### Core Tools

**ingest-pgn** - Import chess games
```bash
./run.sh ingest-pgn --input games.pgn --output data/positions.db
```

**train-cnn** - Train the neural network
```bash
./run.sh train-cnn --dataset data/positions.db --epochs 50
```

**self-improvement** - Autonomous learning
```bash
./run.sh self-improvement --observations 100
```

**live-chess** - Real-time analysis
```bash
./run.sh live-chess --model data/models/chess_cnn.bin
```

**test-model** - Validate model
```bash
./run.sh test-model
```

## Makefile Commands

```bash
make status            # Check system readiness
make demo              # 60-second demo
make quick-start       # Interactive guide
make workflow          # Full pipeline
make test-integration  # Integration tests
make build-tools       # Build all binaries
make run-live-chess    # Run live analysis
make run-self-improve  # Run self-improvement
make help              # Show all commands
```

## Directory Structure

```
P.A.R.T.N.E.R/
├── bin/                  # Compiled binaries
├── cmd/                  # Source code for tools
├── data/                 # Datasets and models
│   ├── models/          # Trained CNN models
│   ├── replays/         # Replay buffer
│   └── *.db             # Position databases
├── internal/            # Core packages
│   ├── model/          # Neural network
│   ├── vision/         # Computer vision
│   ├── training/       # Training system
│   └── data/           # Data management
├── scripts/            # Utility scripts
└── logs/               # Application logs
```

## Troubleshooting

### Go 1.25 Runtime Error

If you see `ASSUME_NO_MOVING_GC_UNSAFE_RISK_IT_WITH` errors:

The `./run.sh` script automatically sets this. If running manually:
```bash
export ASSUME_NO_MOVING_GC_UNSAFE_RISK_IT_WITH=go1.25
```

### OpenCV Not Found

```bash
# Check installation
pkg-config --modversion opencv4

# If missing, reinstall
sudo apt-get install libopencv-dev  # Ubuntu
brew install opencv                  # macOS
```

### Segmentation Fault in Vision

Ensure OpenCV is properly installed and the capture region is valid:
```bash
./run.sh live-chess --x 0 --y 0 --width 800 --height 800
```

### Low Training Accuracy

- Increase epochs: `--epochs 100`
- Use more data (1000+ positions recommended)
- Adjust learning rate: `--learning-rate 0.0001`
- Increase batch size: `--batch-size 64`

### Out of Memory

- Reduce batch size: `--batch-size 16`
- Process smaller PGN files
- Close other applications

## Next Steps

1. **Collect More Data**: More training examples = better accuracy
2. **Experiment with Hyperparameters**: Try different epochs, batch sizes, learning rates
3. **Continuous Improvement**: Run self-improvement regularly
4. **Live Analysis**: Use during actual games for suggestions

## Support

- Check logs in `logs/` directory
- Run `make status` to verify system state
- Run `make test-integration` to validate installation
- Review README.md for detailed architecture

## Examples

### Quick Test with Small Dataset
```bash
# Get a small PGN (100 games)
head -n 2000 large_database.pgn > small_test.pgn

# Ingest and train
./run.sh ingest-pgn --input small_test.pgn --output data/test.db
./run.sh train-cnn --dataset data/test.db --epochs 20 --batch-size 16
```

### Production Training
```bash
# Large dataset, long training
./run.sh ingest-pgn --input lichess_large.pgn --output data/positions.db
./run.sh train-cnn --dataset data/positions.db --epochs 200 --batch-size 64
```

### Continuous Learning Loop
```bash
# Train, improve, repeat
while true; do
    ./run.sh train-cnn --dataset data/positions.db --epochs 50
    ./run.sh self-improvement --observations 200
    sleep 300
done
```

---

**Ready to start?** Run `make demo` for a quick walkthrough!
