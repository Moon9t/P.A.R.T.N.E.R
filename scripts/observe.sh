#!/bin/bash
# Convenience script for observation mode

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

export ASSUME_NO_MOVING_GC_UNSAFE_RISK_IT_WITH=go1.25

./bin/observe-train -mode=observe "$@"
