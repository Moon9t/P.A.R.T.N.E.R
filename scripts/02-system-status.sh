#!/bin/bash
# System Status Checker - Verify P.A.R.T.N.E.R installation and readiness

export ASSUME_NO_MOVING_GC_UNSAFE_RISK_IT_WITH=go1.25

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}╔═══════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║${NC}  P.A.R.T.N.E.R System Status Check                      ${BLUE}║${NC}"
echo -e "${BLUE}╚═══════════════════════════════════════════════════════════╝${NC}"
echo ""

# Check Go installation
echo -n "Go Installation: "
if command -v go &> /dev/null; then
    GO_VERSION=$(go version | awk '{print $3}')
    echo -e "${GREEN}✓${NC} $GO_VERSION"
else
    echo -e "${RED}✗ Not found${NC}"
fi

# Check OpenCV
echo -n "OpenCV: "
if pkg-config --exists opencv4 2>/dev/null; then
    OPENCV_VERSION=$(pkg-config --modversion opencv4)
    echo -e "${GREEN}✓${NC} $OPENCV_VERSION"
else
    echo -e "${RED}✗ Not found${NC}"
fi

# Check build status
echo ""
echo -e "${BLUE}Build Status:${NC}"
TOOLS="ingest-pgn train-cnn test-model self-improvement live-chess"
for tool in $TOOLS; do
    echo -n "  $tool: "
    if [ -f "bin/$tool" ]; then
        echo -e "${GREEN}✓ Built${NC}"
    else
        echo -e "${YELLOW}⚠ Not built${NC}"
    fi
done

# Check data directories
echo ""
echo -e "${BLUE}Data Directories:${NC}"
DIRS="data data/models data/replays logs"
for dir in $DIRS; do
    echo -n "  $dir: "
    if [ -d "$dir" ]; then
        echo -e "${GREEN}✓ Exists${NC}"
    else
        echo -e "${YELLOW}⚠ Missing${NC}"
    fi
done

# Check for datasets
echo ""
echo -e "${BLUE}Datasets:${NC}"
if [ -f "data/positions.db" ]; then
    SIZE=$(du -h data/positions.db | cut -f1)
    echo -e "  positions.db: ${GREEN}✓${NC} $SIZE"
else
    echo -e "  positions.db: ${YELLOW}⚠ Not found${NC}"
fi

# Check for trained models
echo ""
echo -e "${BLUE}Trained Models:${NC}"
if [ -f "data/models/chess_cnn.bin" ]; then
    SIZE=$(du -h data/models/chess_cnn.bin | cut -f1)
    echo -e "  chess_cnn.bin: ${GREEN}✓${NC} $SIZE"
else
    echo -e "  chess_cnn.bin: ${YELLOW}⚠ Not found${NC}"
fi

# Check replay buffer
echo ""
echo -e "${BLUE}Replay Buffer:${NC}"
if [ -f "data/replays/replay.db" ]; then
    SIZE=$(du -h data/replays/replay.db | cut -f1)
    echo -e "  replay.db: ${GREEN}✓${NC} $SIZE"
else
    echo -e "  replay.db: ${YELLOW}⚠ Not found${NC}"
fi

# System readiness
echo ""
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}System Readiness:${NC}"

READY=true

if ! command -v go &> /dev/null; then
    echo -e "${RED}✗ Go not installed${NC}"
    READY=false
fi

if ! pkg-config --exists opencv4 2>/dev/null; then
    echo -e "${RED}✗ OpenCV not installed${NC}"
    READY=false
fi

if [ ! -f "bin/ingest-pgn" ]; then
    echo -e "${YELLOW}⚠ Tools not built - run: make build-tools${NC}"
    READY=false
fi

if $READY; then
    echo -e "${GREEN}✓ System ready!${NC}"
    echo ""
    echo "Next steps:"
    echo "  1. Run: ${GREEN}make quick-start${NC} for interactive guide"
    echo "  2. Or:  ${GREEN}make workflow${NC} to run full pipeline"
    echo "  3. Or:  ${GREEN}./run.sh <tool>${NC} to run specific tools"
else
    echo -e "${RED}✗ System not ready${NC}"
    echo ""
    echo "Required actions:"
    if ! command -v go &> /dev/null; then
        echo "  - Install Go: https://go.dev/doc/install"
    fi
    if ! pkg-config --exists opencv4 2>/dev/null; then
        echo "  - Install OpenCV: see README.md"
    fi
    if [ ! -f "bin/ingest-pgn" ]; then
        echo "  - Build tools: make build-tools"
    fi
fi

echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
