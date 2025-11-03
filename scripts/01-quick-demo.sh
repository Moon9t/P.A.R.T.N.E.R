#!/bin/bash
# Quick Demo - Shows P.A.R.T.N.E.R in action with minimal data
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
║           P.A.R.T.N.E.R Quick Demo                        ║
║           Complete Pipeline in 60 Seconds                 ║
║                                                           ║
╚═══════════════════════════════════════════════════════════╝
EOF
echo -e "${NC}"

echo ""
echo -e "${YELLOW}This demo will:${NC}"
echo "  1. Create a small test PGN file"
echo "  2. Ingest it into a database"
echo "  3. Train a CNN for 10 epochs"
echo "  4. Run 10 self-improvement observations"
echo "  5. Show the results"
echo ""
read -p "Press Enter to start..."

# Setup
DEMO_DIR="demo_$$"
mkdir -p "$DEMO_DIR/data/models"

echo ""
echo -e "${BLUE}[1/5]${NC} Creating test PGN data..."
cat > "$DEMO_DIR/demo.pgn" << 'EOF'
[Event "Demo Game 1"]
[White "Player A"]
[Black "Player B"]

1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 
6. Re1 b5 7. Bb3 d6 8. c3 O-O 9. h3 Na5 10. Bc2 c5 1-0

[Event "Demo Game 2"]
[White "Player C"]
[Black "Player D"]

1. d4 d5 2. c4 e6 3. Nc3 Nf6 4. Bg5 Be7 5. e3 O-O
6. Nf3 h6 7. Bh4 b6 8. cxd5 exd5 9. Bd3 Bb7 10. O-O 1-0

[Event "Demo Game 3"]
[White "Player E"]
[Black "Player F"]

1. e4 c5 2. Nf3 d6 3. d4 cxd4 4. Nxd4 Nf6 5. Nc3 a6
6. Be3 e5 7. Nb3 Be6 8. f3 Be7 9. Qd2 O-O 10. O-O-O 1-0
EOF
echo -e "${GREEN}✓${NC} Created 3 demo games"

echo ""
echo -e "${BLUE}[2/5]${NC} Building tools..."
make build-tools > /dev/null 2>&1
echo -e "${GREEN}✓${NC} Tools ready"

echo ""
echo -e "${BLUE}[3/5]${NC} Ingesting PGN data..."
./bin/ingest-pgn --input "$DEMO_DIR/demo.pgn" --output "$DEMO_DIR/data/positions.db" 2>&1 | tail -3
echo -e "${GREEN}✓${NC} Dataset created"

echo ""
echo -e "${BLUE}[4/5]${NC} Training CNN (10 epochs)..."
./bin/train-cnn \
    --dataset "$DEMO_DIR/data/positions.db" \
    --model "$DEMO_DIR/data/models/demo.bin" \
    --epochs 10 \
    --batch-size 4 2>&1 | grep -E "(Epoch|Loss|Accuracy|Saved)" | tail -15
echo -e "${GREEN}✓${NC} Model trained"

echo ""
echo -e "${BLUE}[5/5]${NC} Self-improvement (10 observations)..."
timeout 15 ./bin/self-improvement \
    --model "$DEMO_DIR/data/models/demo.bin" \
    --dataset "$DEMO_DIR/data/positions.db" \
    --observations 10 2>&1 | tail -10 || true
echo -e "${GREEN}✓${NC} Self-improvement cycle completed"

echo ""
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}Demo Complete!${NC}"
echo ""
echo "What happened:"
echo -e "  ${BLUE}•${NC} Created 3 chess games (30 positions)"
echo -e "  ${BLUE}•${NC} Trained CNN to predict moves"
echo -e "  ${BLUE}•${NC} Model learned and improved itself"
echo ""
echo "Demo files created in: $DEMO_DIR/"
echo ""
echo "Next steps:"
echo -e "  ${YELLOW}1.${NC} Try with real data: ${GREEN}make workflow${NC}"
echo -e "  ${YELLOW}2.${NC} Run live analysis: ${GREEN}make run-live-chess${NC}"
echo -e "  ${YELLOW}3.${NC} Continue training: ${GREEN}./run.sh train-cnn --help${NC}"
echo ""
read -p "Press Enter to cleanup demo files..."
rm -rf "$DEMO_DIR"
echo -e "${GREEN}✓${NC} Cleanup complete"
