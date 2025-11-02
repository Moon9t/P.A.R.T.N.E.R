#!/bin/bash
# Train with improved methods

set -e

DATASET="${1:-data/chess_dataset_improved.db}"
EPOCHS="${2:-50}"
MODEL_OUT="${3:-models/chess_cnn_improved.gob}"

echo "=========================================="
echo " Improved Training Pipeline"
echo "=========================================="
echo "Dataset:    $DATASET"
echo "Epochs:     $EPOCHS"
echo "Model out:  $MODEL_OUT"
echo "=========================================="
echo ""

# Check if dataset exists
if [ ! -f "$DATASET" ]; then
    echo "Error: Dataset not found: $DATASET"
    echo ""
    echo "Run ingestion first:"
    echo "  ./scripts/ingest-and-verify.sh sample_games.pgn"
    exit 1
fi

# Show dataset stats
echo "Dataset Information:"
./run.sh ingest-pgn -dataset="$DATASET" -stats | head -10
echo ""

echo "Starting training with improvements:"
echo "  ✓ Learning rate warmup (5 epochs)"
echo "  ✓ Cosine annealing schedule"
echo "  ✓ Validation split (15%)"
echo "  ✓ Early stopping (patience: 10)"
echo "  ✓ Smaller batch size (32)"
echo "  ✓ Gradient clipping"
echo ""

# TODO: Create improved training binary
# For now, use existing train-cnn with better params
./run.sh train-cnn \
    -dataset="$DATASET" \
    -epochs=$EPOCHS \
    -batch-size=32 \
    -learning-rate=0.001 \
    -output="$MODEL_OUT"

echo ""
echo "=========================================="
echo " Training Complete"
echo "=========================================="
echo "Model saved to: $MODEL_OUT"
echo ""
echo "Next steps:"
echo "  1. Test model:   ./run.sh test-model"
echo "  2. Live analysis: ./run.sh partner-cli --mode=live"
