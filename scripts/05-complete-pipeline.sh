#!/bin/bash
# Full P.A.R.T.N.E.R Workflow - Complete Pipeline
set -e

export ASSUME_NO_MOVING_GC_UNSAFE_RISK_IT_WITH=go1.25

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Helper functions
print_step() {
    echo -e "${BLUE}╔══════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║${NC} ${GREEN}$1${NC}"
    echo -e "${BLUE}╚══════════════════════════════════════════════════════════╝${NC}"
}

print_info() {
    echo -e "${YELLOW}➜${NC} $1"
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

# Configuration
DATASET="data/positions.db"
MODEL="data/models/chess_cnn.bin"
PGN_FILE="${1:-data/sample.pgn}"
EPOCHS="${2:-50}"

print_step "STEP 1: Build All Tools"
print_info "Building P.A.R.T.N.E.R toolchain..."
make build-tools
print_success "All tools built successfully"

print_step "STEP 2: Setup Directories"
print_info "Creating required directories..."
mkdir -p data/models data/replays logs reports
print_success "Directory structure ready"

print_step "STEP 3: Ingest PGN Data"
if [ ! -f "$PGN_FILE" ]; then
    print_error "PGN file not found: $PGN_FILE"
    print_info "Usage: $0 <pgn-file> [epochs]"
    exit 1
fi

print_info "Ingesting chess games from $PGN_FILE..."
./bin/ingest-pgn --input "$PGN_FILE" --output "$DATASET"
print_success "Dataset created: $DATASET"

print_step "STEP 4: Train CNN Model"
print_info "Training neural network for $EPOCHS epochs..."
./bin/train-cnn \
    --dataset "$DATASET" \
    --model "$MODEL" \
    --epochs "$EPOCHS" \
    --batch-size 32 \
    --learning-rate 0.001
print_success "Model trained and saved: $MODEL"

print_step "STEP 5: Run Self-Improvement"
print_info "Running self-improvement cycle..."
./bin/self-improvement \
    --model "$MODEL" \
    --dataset "$DATASET" \
    --observations 50
print_success "Self-improvement cycle completed"

print_step "STEP 6: Test Model"
print_info "Testing trained model..."
./bin/test-model
print_success "Model test completed"

print_step "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
print_success "Full workflow completed successfully!"
echo ""
echo -e "${BLUE}Next Steps:${NC}"
echo -e "  ${YELLOW}1.${NC} Run live analysis: ${GREEN}make run-live-chess${NC}"
echo -e "  ${YELLOW}2.${NC} Continue training: ${GREEN}./bin/train-cnn --dataset $DATASET --model $MODEL --epochs 100${NC}"
echo -e "  ${YELLOW}3.${NC} Self-improve more: ${GREEN}./bin/self-improvement --observations 200${NC}"
echo ""
print_step "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
