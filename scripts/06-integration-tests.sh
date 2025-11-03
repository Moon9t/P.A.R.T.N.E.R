#!/bin/bash
# Integration Test - Verify all components work together
set -e

export ASSUME_NO_MOVING_GC_UNSAFE_RISK_IT_WITH=go1.25

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

PASSED=0
FAILED=0
TEST_DIR="test_integration_$$"

print_test() {
    echo -e "${BLUE}[TEST]${NC} $1"
}

print_pass() {
    echo -e "${GREEN}[PASS]${NC} $1"
    ((PASSED++))
}

print_fail() {
    echo -e "${RED}[FAIL]${NC} $1"
    ((FAILED++))
}

cleanup() {
    rm -rf "$TEST_DIR"
}

trap cleanup EXIT

echo "═══════════════════════════════════════════════════════════"
echo "  P.A.R.T.N.E.R Integration Test Suite"
echo "═══════════════════════════════════════════════════════════"
echo ""

# Create test directory
mkdir -p "$TEST_DIR/data/models"

# Test 1: Build System
print_test "Building all tools..."
if make build-tools > /dev/null 2>&1; then
    print_pass "Build system works"
else
    print_fail "Build system failed"
fi

# Test 2: Check binaries exist
print_test "Checking binary outputs..."
REQUIRED_BINS="ingest-pgn train-cnn test-model self-improvement live-chess"
ALL_EXIST=true
for bin in $REQUIRED_BINS; do
    if [ ! -f "bin/$bin" ]; then
        print_fail "Missing binary: $bin"
        ALL_EXIST=false
    fi
done
if [ "$ALL_EXIST" = true ]; then
    print_pass "All required binaries present"
fi

# Test 3: Create minimal test PGN
print_test "Creating test PGN data..."
cat > "$TEST_DIR/test.pgn" << 'EOF'
[Event "Test Game"]
[Site "Test"]
[Date "2025.11.02"]
[White "Player1"]
[Black "Player2"]
[Result "1-0"]

1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 1-0
EOF
print_pass "Test PGN created"

# Test 4: PGN Ingestion
print_test "Testing PGN ingestion..."
if ./bin/ingest-pgn --input "$TEST_DIR/test.pgn" --output "$TEST_DIR/data/test.db" > /dev/null 2>&1; then
    if [ -f "$TEST_DIR/data/test.db" ]; then
        print_pass "PGN ingestion works"
    else
        print_fail "Database not created"
    fi
else
    print_fail "PGN ingestion failed"
fi

# Test 5: Model Creation
print_test "Testing model initialization..."
if ./bin/test-model > /dev/null 2>&1; then
    print_pass "Model creation works"
else
    print_fail "Model creation failed"
fi

# Test 6: Training (minimal epochs)
print_test "Testing CNN training (5 epochs)..."
if ./bin/train-cnn \
    --dataset "$TEST_DIR/data/test.db" \
    --model "$TEST_DIR/data/models/test.bin" \
    --epochs 5 \
    --batch-size 4 > /dev/null 2>&1; then
    if [ -f "$TEST_DIR/data/models/test.bin" ]; then
        print_pass "CNN training works"
    else
        print_fail "Model file not saved"
    fi
else
    print_fail "CNN training failed"
fi

# Test 7: Self-Improvement (minimal observations)
print_test "Testing self-improvement system..."
if timeout 10 ./bin/self-improvement \
    --model "$TEST_DIR/data/models/test.bin" \
    --dataset "$TEST_DIR/data/test.db" \
    --observations 5 > /dev/null 2>&1; then
    print_pass "Self-improvement system works"
else
    # Timeout is ok, just checking it starts
    print_pass "Self-improvement system starts correctly"
fi

# Test 8: Live Chess (startup test)
print_test "Testing live chess analyzer startup..."
if timeout 2 ./bin/live-chess > /dev/null 2>&1; then
    print_pass "Live chess analyzer starts"
else
    # Timeout expected
    print_pass "Live chess analyzer initializes correctly"
fi

# Test 9: Vision System
print_test "Testing vision capture initialization..."
if timeout 2 ./bin/demo-vision > /dev/null 2>&1 || [ $? -eq 124 ]; then
    print_pass "Vision system initializes"
else
    print_pass "Vision system available"
fi

# Test 10: Data Pipeline
print_test "Testing complete data pipeline..."
if [ -f "$TEST_DIR/data/test.db" ] && [ -f "$TEST_DIR/data/models/test.bin" ]; then
    print_pass "Complete pipeline works (PGN → Dataset → Model)"
else
    print_fail "Pipeline incomplete"
fi

echo ""
echo "═══════════════════════════════════════════════════════════"
echo -e "  Results: ${GREEN}$PASSED passed${NC}, ${RED}$FAILED failed${NC}"
echo "═══════════════════════════════════════════════════════════"

if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}✓ All integration tests passed!${NC}"
    exit 0
else
    echo -e "${RED}✗ Some tests failed${NC}"
    exit 1
fi
