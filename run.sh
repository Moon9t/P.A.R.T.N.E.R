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
    echo "  partner-cli       - Main CLI interface"
    echo "  test-adapter      - Test game adapter system"
    echo "  train-cnn         - Train CNN model"
    echo "  test-model        - Test model implementation"
    echo "  ingest-pgn        - Ingest PGN chess games"
    echo "  live-analysis     - Live game analysis"
    echo ""
    echo "Examples:"
    echo "  ./run.sh test-adapter"
    echo "  ./run.sh partner-cli --help"
    echo "  ./run.sh train-cnn --epochs=50"
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
