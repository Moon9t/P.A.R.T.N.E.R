#!/bin/bash
# P.A.R.T.N.E.R Quick Start Guide
set -e

export ASSUME_NO_MOVING_GC_UNSAFE_RISK_IT_WITH=go1.25

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

clear
echo -e "${BLUE}"
cat << "EOF"
╔═══════════════════════════════════════════════════════════╗
║                                                           ║
║   P.A.R.T.N.E.R - Chess Learning System                  ║
║   Predictive Adaptive Real-Time Neural Evaluation        ║
║                                                           ║
╚═══════════════════════════════════════════════════════════╝
EOF
echo -e "${NC}"

echo ""
echo -e "${GREEN}Welcome to P.A.R.T.N.E.R!${NC}"
echo ""
echo "This quick start guide will help you get started with the system."
echo ""

# Check if Go is installed
if ! command -v go &> /dev/null; then
    echo -e "${YELLOW}⚠ Warning: Go is not installed${NC}"
    echo "Please install Go 1.21 or higher from https://go.dev"
    exit 1
fi

# Check if OpenCV is installed
if ! pkg-config --exists opencv4 2>/dev/null; then
    echo -e "${YELLOW}⚠ Warning: OpenCV not found${NC}"
    echo "Please install OpenCV 4.x:"
    echo "  Ubuntu/Debian: sudo apt-get install libopencv-dev"
    echo "  macOS: brew install opencv"
    echo ""
fi

echo -e "${BLUE}Choose your workflow:${NC}"
echo ""
echo "  1. Complete workflow (PGN → Train → Self-Improve)"
echo "  2. Just train a model from existing dataset"
echo "  3. Run self-improvement on existing model"
echo "  4. Live chess board analysis"
echo "  5. Test existing model"
echo "  6. Build all tools only"
echo "  7. Exit"
echo ""
read -p "Enter your choice (1-7): " choice

case $choice in
    1)
        echo ""
        read -p "Enter path to PGN file: " pgn_file
        if [ ! -f "$pgn_file" ]; then
            echo -e "${YELLOW}File not found. Using default sample if available.${NC}"
            pgn_file="data/sample.pgn"
        fi
        read -p "Number of training epochs (default 50): " epochs
        epochs=${epochs:-50}
        ./scripts/full-workflow.sh "$pgn_file" "$epochs"
        ;;
    2)
        echo ""
        read -p "Dataset path (default data/positions.db): " dataset
        dataset=${dataset:-data/positions.db}
        read -p "Model output path (default data/models/chess_cnn.bin): " model
        model=${model:-data/models/chess_cnn.bin}
        read -p "Number of epochs (default 50): " epochs
        epochs=${epochs:-50}
        
        echo -e "${BLUE}Building tools...${NC}"
        make build-tools
        echo -e "${BLUE}Training model...${NC}"
        ./bin/train-cnn --dataset "$dataset" --model "$model" --epochs "$epochs" --batch-size 32
        ;;
    3)
        echo ""
        read -p "Model path (default data/models/chess_cnn.bin): " model
        model=${model:-data/models/chess_cnn.bin}
        read -p "Dataset path (default data/positions.db): " dataset
        dataset=${dataset:-data/positions.db}
        read -p "Number of observations (default 100): " obs
        obs=${obs:-100}
        
        echo -e "${BLUE}Building tools...${NC}"
        make build-tools
        echo -e "${BLUE}Running self-improvement...${NC}"
        ./bin/self-improvement --model "$model" --dataset "$dataset" --observations "$obs"
        ;;
    4)
        echo ""
        read -p "Model path (default data/models/chess_cnn.bin): " model
        model=${model:-data/models/chess_cnn.bin}
        echo ""
        echo -e "${YELLOW}Position your chess board on screen and press Enter to start...${NC}"
        read
        
        echo -e "${BLUE}Building tools...${NC}"
        make build-tools
        echo -e "${BLUE}Starting live analysis...${NC}"
        echo -e "${YELLOW}Press Ctrl+C to stop${NC}"
        echo ""
        ./bin/live-chess --model "$model"
        ;;
    5)
        echo ""
        echo -e "${BLUE}Building tools...${NC}"
        make build-tools
        echo -e "${BLUE}Testing model...${NC}"
        ./bin/test-model
        ;;
    6)
        echo ""
        echo -e "${BLUE}Building all tools...${NC}"
        make build-tools
        echo -e "${GREEN}✓ Build complete!${NC}"
        echo ""
        echo "Available tools in bin/:"
        ls -1 bin/ | grep -v "\.log" | sed 's/^/  - /'
        ;;
    7)
        echo "Goodbye!"
        exit 0
        ;;
    *)
        echo -e "${YELLOW}Invalid choice${NC}"
        exit 1
        ;;
esac

echo ""
echo -e "${GREEN}Done!${NC}"
