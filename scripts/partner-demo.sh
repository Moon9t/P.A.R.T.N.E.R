#!/bin/bash

# Phase 5/6 Demo Script - CLI Interface

cd "$(dirname "$0")/.."

export ASSUME_NO_MOVING_GC_UNSAFE_RISK_IT_WITH=go1.25

echo "═══════════════════════════════════════════════════════════"
echo "  P.A.R.T.N.E.R Demo"
echo "  Production CLI Interface"
echo "═══════════════════════════════════════════════════════════"
echo ""

# Check if binary exists
if [ ! -f "bin/partner-cli" ]; then
    echo "Building partner-cli..."
    go build -o bin/partner-cli ./cmd/partner-cli || exit 1
    echo "✓ Build complete"
    echo ""
fi

# Show available modes
echo "Available Modes:"
echo "───────────────────────────────────────────────────────────"
echo "  1. observe - Real-time move prediction"
echo "  2. train   - Train model on dataset"
echo "  3. analyze - Test model accuracy"
echo ""

# Parse mode argument or prompt
MODE="${1:-analyze}"

case "$MODE" in
    "observe")
        echo "Starting Observe Mode..."
        echo "Press Ctrl+C to stop"
        echo ""
        ./bin/partner-cli -mode=observe "$@"
        ;;
    
    "train")
        EPOCHS="${2:-10}"
        echo "Starting Training Mode ($EPOCHS epochs)..."
        echo ""
        ./bin/partner-cli -mode=train -epochs="$EPOCHS" "$@"
        ;;
    
    "analyze")
        echo "Starting Analysis Mode..."
        echo ""
        ./bin/partner-cli -mode=analyze "$@"
        ;;
    
    *)
        echo "❌ Unknown mode: $MODE"
        echo ""
        echo "Usage:"
        echo "  $0 observe      - Watch and predict moves"
        echo "  $0 train [N]    - Train for N epochs (default: 10)"
        echo "  $0 analyze      - Test model accuracy"
        exit 1
        ;;
esac

echo ""
echo "═══════════════════════════════════════════════════════════"
