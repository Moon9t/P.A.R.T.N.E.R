#!/bin/bash
# Next Steps - Get P.A.R.T.N.E.R Production Ready
set -e

export ASSUME_NO_MOVING_GC_UNSAFE_RISK_IT_WITH=go1.25

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

clear
echo -e "${BLUE}"
cat << "EOF"
╔═══════════════════════════════════════════════════════════╗
║                                                           ║
║         P.A.R.T.N.E.R - Next Steps Guide                  ║
║                                                           ║
╚═══════════════════════════════════════════════════════════╝
EOF
echo -e "${NC}"

echo ""
echo -e "${CYAN}What comes next?${NC}"
echo ""
echo "Now that the system is integrated, here's your path to a"
echo "production-ready chess AI:"
echo ""

echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}PHASE 1: Get Quality Training Data${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo "Options:"
echo "  1. Download from Lichess Database (recommended)"
echo "  2. Use existing PGN file"
echo "  3. Skip (use demo data)"
echo ""
read -p "Choose (1-3): " data_choice

case $data_choice in
    1)
        echo ""
        echo -e "${YELLOW}Downloading sample from Lichess...${NC}"
        echo "Full database: https://database.lichess.org/"
        echo ""
        echo "For now, creating a small sample..."
        
        # Create sample PGN with more games
        cat > data/sample_training.pgn << 'PGEOF'
[Event "Rated Blitz"]
[White "PlayerA"]
[Black "PlayerB"]
[Result "1-0"]

1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 6. Re1 b5 
7. Bb3 d6 8. c3 O-O 9. h3 Na5 10. Bc2 c5 11. d4 Qc7 12. Nbd2 1-0

[Event "Rated Blitz"]
[White "PlayerC"]
[Black "PlayerD"]
[Result "0-1"]

1. d4 d5 2. c4 e6 3. Nc3 Nf6 4. Bg5 Be7 5. e3 O-O 6. Nf3 h6 
7. Bh4 b6 8. cxd5 exd5 9. Bd3 Bb7 10. O-O Nbd7 0-1

[Event "Rated Blitz"]
[White "PlayerE"]
[Black "PlayerF"]
[Result "1-0"]

1. e4 c5 2. Nf3 d6 3. d4 cxd4 4. Nxd4 Nf6 5. Nc3 a6 6. Be3 e5 
7. Nb3 Be6 8. f3 Be7 9. Qd2 O-O 10. O-O-O Nbd7 1-0
PGEOF
        
        PGN_FILE="data/sample_training.pgn"
        echo -e "${GREEN}✓${NC} Sample data created"
        ;;
    2)
        echo ""
        read -p "Enter path to your PGN file: " user_pgn
        if [ -f "$user_pgn" ]; then
            PGN_FILE="$user_pgn"
            echo -e "${GREEN}✓${NC} Using: $PGN_FILE"
        else
            echo -e "${YELLOW}File not found, using demo data${NC}"
            PGN_FILE="data/sample_training.pgn"
        fi
        ;;
    *)
        echo -e "${YELLOW}Using demo data${NC}"
        PGN_FILE="data/sample_training.pgn"
        ;;
esac

echo ""
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}PHASE 2: Ingest Training Data${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo "Importing PGN into database..."
./run.sh ingest-pgn --input "$PGN_FILE" --output data/positions.db
echo -e "${GREEN}✓${NC} Data imported"

echo ""
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}PHASE 3: Train Initial Model${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
read -p "Number of epochs (default 50): " epochs
epochs=${epochs:-50}

echo ""
echo "Training CNN for $epochs epochs..."
echo "(This may take a few minutes)"
echo ""
./run.sh train-cnn \
    --dataset data/positions.db \
    --model data/models/chess_cnn.bin \
    --epochs "$epochs" \
    --batch-size 32

echo ""
echo -e "${GREEN}✓${NC} Model trained and saved"

echo ""
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}PHASE 4: Test Model${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo "Running model tests..."
./run.sh test-model
echo -e "${GREEN}✓${NC} Model validated"

echo ""
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}PHASE 5: Self-Improvement${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
read -p "Run self-improvement? (y/n): " run_improve

if [[ "$run_improve" =~ ^[Yy]$ ]]; then
    echo ""
    read -p "Number of observations (default 50): " obs
    obs=${obs:-50}
    
    echo ""
    echo "Running self-improvement..."
    ./run.sh self-improvement \
        --model data/models/chess_cnn.bin \
        --dataset data/positions.db \
        --observations "$obs"
    
    echo -e "${GREEN}✓${NC} Self-improvement cycle completed"
fi

echo ""
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${CYAN}✅ SETUP COMPLETE!${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo "Your P.A.R.T.N.E.R system is now ready!"
echo ""
echo -e "${YELLOW}What's next?${NC}"
echo ""
echo "1. ${GREEN}Try Live Analysis${NC}"
echo "   $ make run-live-chess"
echo ""
echo "2. ${GREEN}Train with More Data${NC}"
echo "   $ ./run.sh train-cnn --dataset data/positions.db --epochs 200"
echo ""
echo "3. ${GREEN}Continuous Learning${NC}"
echo "   $ ./run.sh self-improvement --observations 500"
echo ""
echo "4. ${GREEN}Improve the Model${NC}"
echo "   See ROADMAP.md for:"
echo "   - Batch normalization"
echo "   - Deeper architecture"
echo "   - Better augmentation"
echo ""
echo "5. ${GREEN}Check Status${NC}"
echo "   $ make status"
echo ""
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo "Full roadmap available in: ${CYAN}ROADMAP.md${NC}"
echo ""
read -p "Press Enter to exit..."
