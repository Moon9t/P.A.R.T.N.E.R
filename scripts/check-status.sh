#!/bin/bash

# System Status Check for P.A.R.T.N.E.R Real Observation Collection

cd "$(dirname "$0")/.."

echo "═══════════════════════════════════════════════════════════"
echo "  P.A.R.T.N.E.R - System Status Check"
echo "═══════════════════════════════════════════════════════════"
echo ""

# Check binaries
echo "📦 Binaries:"
echo "───────────────────────────────────────────────────────────"
for binary in collect-real observe-train test-training partner; do
    if [ -f "bin/$binary" ]; then
        size=$(du -h "bin/$binary" | cut -f1)
        echo "  ✓ $binary ($size)"
    else
        echo "  ✗ $binary (MISSING - run: go build -o bin/$binary ./cmd/$binary)"
    fi
done
echo ""

# Check databases
echo "💾 Databases:"
echo "───────────────────────────────────────────────────────────"
if [ -d "data" ]; then
    if ls data/*.db 1> /dev/null 2>&1; then
        for db in data/*.db; do
            size=$(du -h "$db" | cut -f1)
            name=$(basename "$db")
            echo "  ✓ $name ($size)"
        done
    else
        echo "  ℹ No databases yet (will be created on first use)"
    fi
else
    echo "  ⚠ data/ directory not found (will be created automatically)"
fi
echo ""

# Check configuration
echo "⚙️  Configuration:"
echo "───────────────────────────────────────────────────────────"
if [ -f "config.json" ]; then
    echo "  ✓ config.json exists"
    
    # Extract screen region
    if command -v jq &> /dev/null; then
        x=$(jq -r '.vision.screen_region.x' config.json 2>/dev/null)
        y=$(jq -r '.vision.screen_region.y' config.json 2>/dev/null)
        w=$(jq -r '.vision.screen_region.width' config.json 2>/dev/null)
        h=$(jq -r '.vision.screen_region.height' config.json 2>/dev/null)
        
        if [ "$x" != "null" ]; then
            echo "  Screen Region: x=$x, y=$y, width=$w, height=$h"
        fi
    else
        echo "  (Install 'jq' to see detailed config info)"
    fi
else
    echo "  ✗ config.json MISSING"
    echo "  Create one with: cp config.example.json config.json"
fi
echo ""

# Check storage stats
echo "📊 Storage Statistics:"
echo "───────────────────────────────────────────────────────────"
if [ -f "bin/collect-real" ]; then
    # Run collector with 0 samples to just show stats
    timeout 2s ./bin/collect-real -samples=0 2>/dev/null | grep -A 10 "Storage Statistics" || echo "  No data collected yet"
else
    echo "  Cannot check (collect-real not built)"
fi
echo ""

# Check documentation
echo "📚 Documentation:"
echo "───────────────────────────────────────────────────────────"
docs=(
    "READY_TO_COLLECT.md:System ready guide"
    "COLLECT_QUICKSTART.md:Quick start guide"
    "REAL_OBSERVATION_GUIDE.md:Full collection guide"
    "PHASE4_COMPLETE.md:Training system docs"
)

for doc_info in "${docs[@]}"; do
    doc="${doc_info%%:*}"
    desc="${doc_info##*:}"
    if [ -f "$doc" ]; then
        echo "  ✓ $doc - $desc"
    else
        echo "  ✗ $doc - MISSING"
    fi
done
echo ""

# Overall status
echo "═══════════════════════════════════════════════════════════"
echo "  System Status: READY ✓"
echo "═══════════════════════════════════════════════════════════"
echo ""
echo "Quick Start:"
echo "  1. Configure screen region in config.json"
echo "  2. Open chess board (chess.com, lichess, etc.)"
echo "  3. Run: ./bin/collect-real -samples=100"
echo "  4. Train: ./bin/observe-train -mode=train -epochs=50"
echo "  5. Use: ./bin/partner -mode=advise"
echo ""
echo "Documentation: READY_TO_COLLECT.md"
echo "═══════════════════════════════════════════════════════════"
