#!/bin/bash
# Training Data Collection Wrapper
# Automatically collects chess game data for model training

echo "════════════════════════════════════════════════════════════"
echo "  P.A.R.T.N.E.R - Training Data Collection (Priority 2)"
echo "════════════════════════════════════════════════════════════"
echo ""
echo "This tool uses ASYNC CAPTURE (4.8x faster) to collect training data"
echo ""
echo "What it does:"
echo "  • Monitors your chess board in real-time"
echo "  • Detects moves automatically"
echo "  • Stores board states + moves in database"
echo "  • Uses async capture for better performance"
echo ""
echo "Prerequisites:"
echo "  1. Open a chess game (lichess.org, chess.com, etc.)"
echo "  2. Ensure board is visible in configured screen region"
echo "  3. Play or watch games normally"
echo ""
echo "Default: Collect 1000 samples at 10 FPS"
echo "Press Ctrl+C anytime to stop and save"
echo ""
echo "Starting in 3 seconds..."
sleep 3

cd /home/thyrook/Desktop/P.A.R.T.N.E.R
ASSUME_NO_MOVING_GC_UNSAFE_RISK_IT_WITH=go1.25 ./bin/collect-training-data \
    -samples=1000 \
    -fps=10 \
    -verbose
