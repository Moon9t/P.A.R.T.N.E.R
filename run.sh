#!/bin/bash
# Wrapper script to run P.A.R.T.N.E.R binaries with required environment variable

export ASSUME_NO_MOVING_GC_UNSAFE_RISK_IT_WITH=go1.25

# If no arguments provided, show usage
if [ $# -eq 0 ]; then
    echo "P.A.R.T.N.E.R Launcher"
    echo "===================="
    echo ""
    echo "Usage: ./run.sh <binary> [arguments]"
    echo ""
    echo "Available binaries:"
    ls -1 bin/ 2>/dev/null | grep -v "\.log" | sed 's/^/  - /'
    echo ""
    echo "Examples:"
    echo "  ./run.sh ingest-pgn --input games.pgn --output data/positions.db"
    echo "  ./run.sh train-cnn --dataset data/positions.db --epochs 50"
    echo "  ./run.sh self-improvement --observations 100"
    echo "  ./run.sh live-chess --model data/models/chess_cnn.bin"
    echo "  ./run.sh test-model"
    exit 0
fi

BINARY=$1
shift

# Check if binary exists
if [ ! -f "bin/$BINARY" ]; then
    echo "Error: Binary 'bin/$BINARY' not found"
    echo "Run 'make build-tools' to build all binaries"
    exit 1
fi

# Run the binary with remaining arguments
exec "bin/$BINARY" "$@"
