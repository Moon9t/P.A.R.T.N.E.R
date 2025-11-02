#!/bin/bash

# Phase 5 Demo Script - Decision Engine & Enhanced UI

cd "$(dirname "$0")/.."

export ASSUME_NO_MOVING_GC_UNSAFE_RISK_IT_WITH=go1.25

echo "═══════════════════════════════════════════════════════════"
echo "  P.A.R.T.N.E.R Phase 5 Demo"
echo "  Decision Engine & Enhanced UI"
echo "═══════════════════════════════════════════════════════════"
echo ""

# Check if binary exists
if [ ! -f "bin/partner-v2" ]; then
    echo "Building partner-v2..."
    go build -o bin/partner-v2 ./cmd/partner-v2 || exit 1
    echo "✓ Build complete"
    echo ""
fi

# Show available modes
echo "Available Modes:"
echo "───────────────────────────────────────────────────────────"
echo "  1. assist  - Real-time chess assistance with decision engine"
echo "  2. train   - Train model on collected observations"
echo "  3. collect - Collect observations from live gameplay"
echo "  4. stats   - Display system statistics"
echo ""

# Parse mode argument or prompt
MODE="${1:-stats}"

case "$MODE" in
    "assist")
        echo "Starting Assistance Mode..."
        echo "Press Ctrl+C to stop"
        echo ""
        ./bin/partner-v2 -mode=assist -tts=false "$@"
        ;;
    
    "train")
        EPOCHS="${2:-50}"
        echo "Starting Training Mode ($EPOCHS epochs)..."
        echo ""
        ./bin/partner-v2 -mode=train -epochs="$EPOCHS" "$@"
        ;;
    
    "collect")
        NUM="${2:-100}"
        echo "Starting Collection Mode ($NUM samples)..."
        echo ""
        ./bin/partner-v2 -mode=collect -collect="$NUM" "$@"
        ;;
    
    "stats")
        ./bin/partner-v2 -mode=stats "$@"
        ;;
    
    *)
        echo "❌ Unknown mode: $MODE"
        echo ""
        echo "Usage:"
        echo "  $0 assist     - Start assistance mode"
        echo "  $0 train [N]  - Train for N epochs (default: 50)"
        echo "  $0 collect [N]- Collect N samples (default: 100)"
        echo "  $0 stats      - Show statistics"
        exit 1
        ;;
esac

echo ""
echo "═══════════════════════════════════════════════════════════"
