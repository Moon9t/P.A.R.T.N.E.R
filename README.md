# P.A.R.T.N.E.R

**P**redictive **A**daptive **R**eal-**T**ime **N**eural **E**valuation & **R**esponse

A production-grade AI system written entirely in Go that learns from chess gameplay through behavioral cloning and provides real-time move suggestions using convolutional neural networks.

## Quick Start

Get started in 3 commands:

```bash
# 1. Build all tools
make build

# 2. Ingest training data from PGN files
./run.sh ingest-pgn --input data/sample_games.pgn --output data/positions.db

# 3. Train the CNN model
./run.sh train-cnn --dataset data/positions.db --model data/models/chess_model.gob --epochs 50
```

For manual control, use the universal runner:

```bash
./run.sh <tool-name> [args]
```

## Features

- **Pure Go Implementation**: No Python dependencies, fully implemented in Go
- **CPU-Optimized**: Runs efficiently on modest hardware (Intel i5 6th Gen, 8GB RAM)
- **Computer Vision**: Real-time screen capture and board state detection using GoCV
- **Deep Learning**: Convolutional neural network built with Gorgonia
- **PGN Import**: Import chess games from standard PGN files
- **Behavioral Cloning**: Learns from master-level gameplay
- **Persistent Storage**: Uses BoltDB for efficient position storage
- **Real-Time Analysis**: Live board detection and move prediction
- **Multiple Tools**: CLI, training, ingestion, and analysis tools

## Architecture

```
P.A.R.T.N.E.R/
├── cmd/
│   ├── partner-cli/     # Interactive CLI tool
│   ├── train-cnn/       # CNN training tool
│   ├── ingest-pgn/      # PGN import tool
│   ├── live-chess/      # Live board analysis
│   └── live-analysis/   # Real-time analysis engine
├── internal/
│   ├── adapter/         # Game adapter system
│   ├── config/          # Configuration management
│   ├── data/            # Dataset and PGN parsing
│   ├── decision/        # Decision engine and move ranking
│   ├── iface/           # CLI and logging
│   ├── model/           # CNN architecture and training
│   ├── storage/         # BoltDB observation storage
│   ├── training/        # Training and replay buffer
│   └── vision/          # Computer vision and board detection
├── bin/                 # Compiled binaries
├── data/                # Training data and models
│   ├── positions.db     # BoltDB position database
│   └── models/          # Trained CNN models
├── logs/                # Application logs
├── configs/             # Configuration files
├── go.mod              # Go module definition
├── Makefile            # Build automation
├── run.sh              # Universal runner script
└── README.md           # This file
```

## Prerequisites

### System Requirements

- **OS**: Linux (Ubuntu 18.04+), macOS 10.13+
- **CPU**: Intel i5 6th Gen or better
- **RAM**: 8GB minimum
- **Go**: 1.21 or higher (tested with Go 1.25)

### Dependencies

#### 1. Go Installation

```bash
# Download and install Go 1.21+
wget https://go.dev/dl/go1.21.0.linux-amd64.tar.gz
sudo tar -C /usr/local -xzf go1.21.0.linux-amd64.tar.gz
export PATH=$PATH:/usr/local/go/bin
```

#### 2. OpenCV Installation

GoCV requires OpenCV 4.x. Install it based on your platform:

**Ubuntu/Debian:**

```bash
sudo apt-get update
sudo apt-get install -y \
    libopencv-dev \
    libopencv-contrib-dev \
    pkg-config
```

**macOS:**

```bash
brew install opencv
brew install pkgconfig
```

**Or build from source:**

```bash
# Clone GoCV and run the installation script
cd /tmp
git clone https://github.com/hybridgroup/gocv.git
cd gocv
make install
```

#### 3. Additional Libraries

```bash
# X11 libraries for screen capture (Linux only)
sudo apt-get install -y libx11-dev xorg-dev libxtst-dev

# BoltDB and other Go dependencies (installed automatically)
go get -u go.etcd.io/bbolt
go get -u go.uber.org/zap
```

## Installation

### 1. Clone or Navigate to the Project

```bash
cd /path/to/P.A.R.T.N.E.R
```

### 2. Install Go Dependencies

```bash
go mod download
go mod tidy
```

### 3. Build All Tools

```bash
make build
```

This will build all binaries to the `bin/` directory:
- `partner` - Interactive CLI tool
- `train-cnn` - CNN training utility
- `ingest-pgn` - PGN import tool
- `live-chess` - Live board analysis
- `live-analysis` - Real-time analysis engine

### 4. Create Required Directories

```bash
mkdir -p data/models data/replays logs
```

## Configuration

Edit `configs/partner.json` to customize the system:

```json
{
  "vision": {
    "screen_region": {
      "x": 100,
      "y": 100,
      "width": 640,
      "height": 640
    },
    "capture_fps": 2,
    "board_size": 8,
    "diff_threshold": 15.0
  },
  "model": {
    "input_size": 64,
    "hidden_size": 256,
    "output_size": 4096,
    "learning_rate": 0.001,
    "batch_size": 32,
    "model_path": "data/models/chess_cnn.gob"
  },
  "training": {
    "replay_buffer_size": 10000,
    "db_path": "data/positions.db",
    "save_interval": 100,
    "min_samples_before_training": 50
  },
  "interface": {
    "log_level": "info",
    "log_path": "logs/partner.log",
    "confidence_threshold": 0.3
  }
}
```

### Configuring Screen Capture Region

To determine the correct screen region for the chessboard:

1. Open your chess application
2. Use a tool like `xwininfo` (Linux) to get window coordinates
3. Update the `screen_region` in `configs/partner.json`

```bash
# Linux: Click on the chess window after running this
xwininfo | grep -E "Absolute upper-left|Width|Height"
```

## Usage

P.A.R.T.N.E.R provides multiple tools for different workflows:

### Available Commands

All tools are accessed through the `run.sh` wrapper script:

```bash
./run.sh <tool-name> [flags]
```

### 1. Interactive CLI - partner

Interactive command-line interface for dataset management and training:

```bash
./run.sh partner
```

**Features:**
- Dataset management (view, export, clear)
- Model training
- Inference testing
- Interactive menu system

### 2. PGN Ingestion - ingest-pgn

Import chess games from PGN files into the training database:

```bash
./run.sh ingest-pgn --input <pgn-file> --output <database-file>
```

**Flags:**
- `--input` - Path to PGN file (required)
- `--output` - Path to output database (default: "data/positions.db")

**Example:**
```bash
./run.sh ingest-pgn --input data/sample_games.pgn --output data/positions.db
```

### 3. CNN Training - train-cnn

Train the convolutional neural network on stored positions:

```bash
./run.sh train-cnn --dataset <database> --model <model-file> --epochs <num> --batch-size <size>
```

**Flags:**
- `--dataset` - Path to training dataset (default: "data/chess_dataset.db")
- `--model` - Path to save/load model (default: "models/chess_cnn.gob")
- `--epochs` - Number of training epochs (default: 10)
- `--batch-size` - Batch size for training (default: 64)
- `--lr` - Learning rate (default: 0.001)
- `--load` - Load existing model before training
- `--test` - Test mode: just run inference on a sample

**Example:**
```bash
./run.sh train-cnn --dataset data/positions.db --model data/models/chess_model.gob --epochs 50 --batch-size 64
```

### 4. Live Chess Analysis - live-chess

Real-time board capture and move prediction:

```bash
./run.sh live-chess --model <model-file> --x <x> --y <y> --width <w> --height <h> --fps <fps> --top <k>
```

**Flags:**
- `--model` - Path to trained model (required)
- `--x` - Screen X coordinate (default: 100)
- `--y` - Screen Y coordinate (default: 100)
- `--width` - Capture width (default: 800)
- `--height` - Capture height (default: 800)
- `--fps` - Frames per second (default: 2)
- `--top` - Number of top moves to show (default: 5)

**Example:**
```bash
./run.sh live-chess --model data/models/chess_cnn.gob --x 100 --y 100 --width 800 --height 800
```

### 5. Live Analysis - live-analysis

Advanced real-time analysis with decision engine:

```bash
./run.sh live-analysis
```

**Features:**
- Real-time board detection
- Move ranking with confidence scores
- Pattern detection
- Tactical analysis

## Workflow Example

Here's a complete workflow from setup to getting predictions:

### Step 1: Build All Tools

```bash
make build
```

### Step 2: Prepare Training Data

Download or prepare PGN files with chess games (master-level games recommended):

```bash
# Example: sample games are included
ls data/sample_games.pgn
```

### Step 3: Ingest PGN Data

Import chess games into the training database:

```bash
./run.sh ingest-pgn --input data/sample_games.pgn --output data/positions.db
```

This parses all games and stores board positions with their corresponding moves.

### Step 4: Train the CNN Model

Train the neural network on the imported positions:

```bash
./run.sh train-cnn --dataset data/positions.db --model data/models/chess_model.gob --epochs 50 --batch-size 64
```

**Training Progress:**
```
Chess CNN Training Tool
================
Loading dataset from: data/positions.db
Loaded 1234 positions

Epoch 1/50: Loss=2.3456
Epoch 2/50: Loss=2.1234
...
Epoch 50/50: Loss=0.8765

Training complete! Model saved to: data/models/chess_model.gob
```

### Step 5: Run Live Analysis

Use the trained model for real-time board analysis:

```bash
./run.sh live-chess --model data/models/chess_model.gob --x 100 --y 100 --width 800 --height 800
```

Or use the interactive CLI:

```bash
./run.sh partner
# Choose option 3 for inference
```

## Understanding the Output

### PGN Ingestion Output

```
Parsing PGN file: data/sample_games.pgn
Processed game 1: 45 positions
Processed game 2: 52 positions
Processed game 3: 38 positions
...
Total positions stored: 1234
Database saved to: data/positions.db
```

### Training Output

```
Chess CNN Training Tool
================
Loading dataset from: data/positions.db
Loaded 1234 positions

Epoch 1/50: Loss=2.3456, Accuracy=0.234
Epoch 2/50: Loss=2.1234, Accuracy=0.312
Epoch 10/50: Loss=1.5432, Accuracy=0.456
...
Epoch 50/50: Loss=0.8765, Accuracy=0.723

Training complete! Model saved to: data/models/chess_model.gob
```

### Live Chess Output

```
Live Chess Vision Analysis
=========================
Model loaded: data/models/chess_cnn.gob
Capturing from: (100, 100) 800x800

Board detected. Top 5 moves:
1. e2e4 (87.3%) - Highly recommended. Controls center.
2. d2d4 (72.1%) - Strong candidate. Solid positional play.
3. g1f3 (68.5%) - Reasonable choice. Developing the position.
4. c2c4 (54.2%) - Fair. English opening.
5. e2e3 (45.7%) - Risky. Passive opening.
```

### Interactive CLI Output

```
P.A.R.T.N.E.R Interactive CLI
============================
1. View Dataset Stats
2. Train Model
3. Run Inference
4. Export Data
5. Clear Dataset
6. Exit

Select option:
```

## Advanced Usage

### Makefile Commands

The project includes a comprehensive Makefile for common tasks:

```bash
# Build commands
make build              # Build all tools
make build-tools        # Build with environment variables set
make clean              # Clean build artifacts

# Run commands
make run-live-chess     # Run live chess analysis
make run-self-improve   # Run self-improvement system

# Testing
make test               # Run unit tests
make test-integration   # Run integration tests
make test-adapter       # Test game adapter interface

# Development
make fmt                # Format code
make lint               # Lint code (requires golangci-lint)

# Quick start guides
make status             # Check system status
make demo               # Run 60-second demo
make quick-start        # Interactive setup guide
make workflow           # Complete workflow demo
```

### Custom Screen Regions

For live analysis, specify custom screen regions:

```bash
./run.sh live-chess --model data/models/chess_cnn.gob --x 200 --y 150 --width 1024 --height 1024 --fps 4
```

### Batch Training

Train on multiple PGN files:

```bash
# Ingest multiple files
./run.sh ingest-pgn --input data/training/Hikaru_games.pgn --output data/positions.db
./run.sh ingest-pgn --input data/training/Carlsen_games.pgn --output data/positions.db

# Train on combined dataset
./run.sh train-cnn --dataset data/positions.db --model data/models/chess_model.gob --epochs 100 --batch-size 128
```

### Model Loading

Continue training from existing model:

```bash
./run.sh train-cnn --dataset data/positions.db --model data/models/chess_model.gob --epochs 50 --load
```

## Performance Optimization

### For Low-End Systems

Use smaller batch sizes and fewer epochs:

```bash
./run.sh train-cnn --dataset data/positions.db --model data/models/chess_model.gob --epochs 20 --batch-size 16
```

Reduce capture FPS for live analysis:

```bash
./run.sh live-chess --model data/models/chess_cnn.gob --fps 1
```

### For Better Performance

Use larger batch sizes and more training epochs:

```bash
./run.sh train-cnn --dataset data/positions.db --model data/models/chess_model.gob --epochs 100 --batch-size 128
```

Increase capture rate for faster response:

```bash
./run.sh live-chess --model data/models/chess_cnn.gob --fps 4
```

### GPU Acceleration

Note: Current implementation is CPU-only. Gorgonia supports CUDA, but requires additional setup.

## Troubleshooting

### Issue: "flag provided but not defined"

**Solution:**
Check the available flags for each tool. Common mistake is using `--output` instead of `--model` for train-cnn.

Correct flags:
- ingest-pgn: `--input`, `--output`
- train-cnn: `--dataset`, `--model`, `--epochs`, `--batch-size`, `--lr`, `--load`, `--test`
- live-chess: `--model`, `--x`, `--y`, `--width`, `--height`, `--fps`, `--top`

### Issue: "Failed to capture screen"

**Solution:**
- Check screen region coordinates with xwininfo (Linux)
- Ensure the chess application is visible and not minimized
- Verify X11 libraries are installed: `sudo apt-get install libx11-dev libxtst-dev`
- On macOS, grant screen recording permissions in System Preferences

### Issue: "Failed to load model" or "no such file or directory"

**Solution:**
- Normal on first run - model doesn't exist yet
- Train the model first: `./run.sh train-cnn --dataset data/positions.db --model data/models/chess_model.gob --epochs 50`
- Check if model file exists: `ls -la data/models/`
- Ensure correct path is specified

### Issue: Low prediction accuracy

**Solution:**
- Import more high-quality games (master-level PGN files)
- Train for more epochs: `--epochs 100` or higher
- Increase batch size if RAM allows: `--batch-size 128`
- Ensure PGN files are properly formatted
- Use games from strong players (2500+ ELO rating)

### Issue: "database is locked" or BoltDB errors

**Solution:**
- Ensure only one process is accessing the database at a time
- Close any running partner CLI instances
- Check for stale lock files in data/ directory
- Restart if necessary

### Issue: High memory usage

**Solution:**
- Reduce batch size: `--batch-size 32` or lower
- Close other applications
- Monitor with: `top` or `htop`
- Consider using a machine with more RAM

### Issue: Compilation errors with Gorgonia

**Solution:**
- Set the environment variable: `export ASSUME_NO_MOVING_GC_UNSAFE_RISK_IT_WITH=go1.25`
- The `run.sh` script sets this automatically
- For manual builds: `ASSUME_NO_MOVING_GC_UNSAFE_RISK_IT_WITH=go1.25 go build ...`

## Command Reference

### run.sh

Universal runner script that sets required environment variables:

```bash
./run.sh <tool> [flags]
```

Automatically sets: `ASSUME_NO_MOVING_GC_UNSAFE_RISK_IT_WITH=go1.25`

### Available Tools

#### partner
Interactive CLI for dataset management, training, and inference.
```bash
./run.sh partner
```

#### ingest-pgn
Import PGN files into training database.
```bash
./run.sh ingest-pgn --input <file.pgn> --output <database.db>
```

#### train-cnn
Train the CNN model on stored positions.
```bash
./run.sh train-cnn --dataset <db> --model <model.gob> --epochs <n> --batch-size <size>
```

#### live-chess
Real-time board capture and move prediction.
```bash
./run.sh live-chess --model <model.gob> --x <x> --y <y> --width <w> --height <h>
```

#### live-analysis
Advanced analysis with decision engine and pattern detection.
```bash
./run.sh live-analysis
```

## Data Sources

### Where to Find PGN Files

High-quality PGN files for training:

1. **lichess.org** - Download game databases by player
   - [https://lichess.org/api](https://lichess.org/api)
   - Filter by rating (2500+ recommended)

2. **pgnmentor.com** - Curated collections of master games
   - Historical games
   - World championship matches

3. **chessgames.com** - Large database of annotated games
   - Download by player, opening, or event

4. **FICS Games Database** - Free Internet Chess Server archives
   - Millions of games available

### Recommended Training Data

For best results:
- 10,000+ positions minimum
- Games from players rated 2400+
- Mix of openings and game types
- Complete games (not fragments)

Example workflow:
```bash
# Download from lichess (example)
wget https://lichess.org/api/games/user/DrNykterstein?max=100 -O hikaru.pgn

# Ingest into database
./run.sh ingest-pgn --input hikaru.pgn --output data/positions.db

# Train model
./run.sh train-cnn --dataset data/positions.db --model data/models/chess_model.gob --epochs 50
```

## System Architecture

### Game Adapter System

P.A.R.T.N.E.R uses an adapter pattern for game-agnostic learning:

- **GameAdapter Interface**: Abstract interface for any game
- **ChessAdapter**: Reference implementation for chess
- **Extensible**: Easy to add new games

See `docs/ADAPTER_SYSTEM.md` for details.

## Technical Implementation

### CNN Architecture

Current implementation uses a convolutional neural network with:
- Input layer: 8x8 board representation (64 squares)
- Convolutional layers with ReLU activation
- Fully connected layers
- Output layer: 4096 possible moves (64x64 from-to combinations)
- Loss function: Cross-entropy
- Optimizer: Adam with cosine annealing learning rate schedule

### Data Pipeline

1. **PGN Parsing** (`internal/data/pgn_parser.go`)
   - Parses standard PGN format
   - Extracts board positions and moves
   - Handles variations and annotations

2. **Tensorization** (`internal/data/tensorize.go`)
   - Converts board states to tensors
   - Encodes moves as integers (0-4095)
   - Normalizes board representation

3. **Storage** (`internal/storage/observation.go`)
   - BoltDB for persistent storage
   - Efficient key-value access
   - JSON export capability

4. **Training** (`internal/model/trainer.go`)
   - Batch loading from database
   - Data augmentation (horizontal flip, color inversion)
   - Learning rate scheduling (warmup + cosine annealing)
   - Model checkpointing

## Performance Benchmarks

On Intel i5 6th Gen, 8GB RAM:

- **Screen Capture**: ~30ms per frame
- **Inference**: ~50-100ms per prediction
- **Training**: ~2-5 seconds per epoch (batch_size=32)
- **Memory Usage**: ~200-500MB (depends on buffer size)

## Future Enhancements

Potential improvements and extensions:

- UCI protocol integration for direct chess engine communication
- Multi-board game support through adapter system
- Distributed training across multiple machines
- Web interface for remote access
- Mobile app integration
- Cloud model storage and sharing
- GPU acceleration via CUDA support in Gorgonia
- Reinforcement learning for self-play
- Opening book integration
- Endgame tablebase support

## Technical Details

### CNN Architecture Details

```
Input: 1x8x8 (grayscale board representation)
  ↓
Conv2D(16 filters, 3x3) + ReLU + MaxPool(2x2)
  ↓
Conv2D(32 filters, 3x3) + ReLU + MaxPool(2x2)
  ↓
Flatten
  ↓
Dense(hidden_size) + ReLU
  ↓
Dense(hidden_size) + ReLU
  ↓
Dense(4096) + Softmax
  ↓
Output: 64x64 move probabilities
```

### Move Encoding

Moves are encoded as integers from 0 to 4095:

- `move_index = from_square * 64 + to_square`
- `from_square = rank * 8 + file` (0-63)
- Example: e2e4 → 12 * 64 + 28 = 796

### Learning Algorithm

1. **Behavioral Cloning**: Learn from (state, action) pairs
2. **Experience Replay**: Store and sample from past observations
3. **Incremental Training**: Update weights gradually
4. **Cross-Entropy Loss**: Minimize prediction error

## License

This project is provided as-is for educational and personal use.

## Acknowledgments

- **Gorgonia**: Go machine learning library
- **GoCV**: Go bindings for OpenCV
- **BoltDB**: Embedded database for Go

## Project Status

Current version: Production-ready MVP

Completed features:
- PGN ingestion and parsing
- CNN training with data augmentation
- Real-time board capture and analysis
- Interactive CLI tool
- Decision engine with pattern detection
- Persistent storage with BoltDB
- Game adapter system

See CHANGELOG.md for version history and ROADMAP.md for planned features.

## Support and Documentation

For issues, questions, or contributions:

- Check the troubleshooting section above
- Review logs in `logs/` directory
- Examine configuration in `configs/partner.json`
- Read documentation in `docs/` directory
- See QUICK_START.md for rapid setup guide

## Related Files

- `CHANGELOG.md` - Version history and changes
- `ROADMAP.md` - Planned features and future development
- `QUICK_START.md` - Fast setup guide
- `LICENSE` - Project license
- `docs/ADAPTER_SYSTEM.md` - Game adapter documentation

---

Built with Go. No Python, no compromise, pure performance.
