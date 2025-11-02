#!/bin/bash

# Quick start script for collecting real chess observations

cd "$(dirname "$0")/.."

export ASSUME_NO_MOVING_GC_UNSAFE_RISK_IT_WITH=go1.25

echo "═══════════════════════════════════════════════════════════"
echo "  P.A.R.T.N.E.R - Real Observation Collection"
echo "═══════════════════════════════════════════════════════════"
echo ""

# Check if binary exists
if [ ! -f "bin/collect-real" ]; then
    echo "Building collect-real..."
    go build -o bin/collect-real ./cmd/collect-real || exit 1
    echo "✓ Build complete"
    echo ""
fi

# Parse arguments or use defaults
SAMPLES="${1:-0}"
FPS="${2:-2}"

if [ "$SAMPLES" -eq 0 ]; then
    echo "Mode: Unlimited collection (Ctrl+C to stop)"
    echo "FPS:  $FPS checks per second"
else
    echo "Mode: Collect $SAMPLES samples"
    echo "FPS:  $FPS checks per second"
fi

echo ""
echo "═══════════════════════════════════════════════════════════"
echo ""

# Run collector
./bin/collect-real -samples="$SAMPLES" -fps="$FPS" "$@"
