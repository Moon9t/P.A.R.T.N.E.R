#!/bin/bash
# Enhanced PGN ingestion with verification

set -e

# Configuration
PGN_FILE="${1:-sample_games.pgn}"
OUTPUT_DB="${2:-data/chess_dataset_improved.db}"
MAX_GAMES="${3:-0}"

echo "=========================================="
echo " Enhanced PGN Ingestion"
echo "=========================================="
echo "Input:  $PGN_FILE"
echo "Output: $OUTPUT_DB"
echo "=========================================="
echo ""

# Remove old database if exists
if [ -f "$OUTPUT_DB" ]; then
    echo "Removing old database..."
    rm "$OUTPUT_DB"
fi

# Count games in PGN
GAME_COUNT=$(grep -c "^\[Event " "$PGN_FILE" || echo "0")
echo "Games in PGN file: $GAME_COUNT"
echo ""

# Run ingestion
echo "Starting ingestion..."
./run.sh ingest-pgn \
    -pgn="$PGN_FILE" \
    -dataset="$OUTPUT_DB" \
    -max-games=$MAX_GAMES \
    -workers=4 \
    -verify

echo ""
echo "=========================================="
echo " Ingestion Complete"
echo "=========================================="
echo ""

# Show detailed statistics
echo "Detailed Statistics:"
./run.sh ingest-pgn -dataset="$OUTPUT_DB" -stats

echo ""
echo "=========================================="
echo " Analysis"
echo "=========================================="

# Get total positions
TOTAL_POSITIONS=$(./run.sh ingest-pgn -dataset="$OUTPUT_DB" -stats 2>/dev/null | grep "Total entries:" | awk '{print $3}')

if [ ! -z "$TOTAL_POSITIONS" ]; then
    POSITIONS_PER_GAME=$((TOTAL_POSITIONS / GAME_COUNT))
    echo "Average positions per game: $POSITIONS_PER_GAME"
    echo ""
    
    if [ $POSITIONS_PER_GAME -lt 10 ]; then
        echo "⚠️  WARNING: Very few positions per game!"
        echo "   Expected: ~25-40 positions per game"
        echo "   Actual:   $POSITIONS_PER_GAME positions per game"
        echo ""
        echo "   This indicates the PGN processor may not be extracting"
        echo "   all positions correctly. Check internal/data/pgn_parser.go"
    elif [ $POSITIONS_PER_GAME -lt 20 ]; then
        echo "⚠️  WARNING: Low positions per game"
        echo "   Consider extracting more positions from each game"
    else
        echo "✓ Good extraction rate!"
        echo "  This looks healthy for training"
    fi
fi

echo ""
echo "Dataset ready: $OUTPUT_DB"
echo "Use: ./run.sh train-cnn -dataset=$OUTPUT_DB"
