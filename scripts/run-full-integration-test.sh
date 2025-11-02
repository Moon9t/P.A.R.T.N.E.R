#!/bin/bash
# Full 5-minute System Integration Test
# Tests all modules: Vision → Storage → Model → Training → Decision

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

echo -e "${BOLD}${CYAN}"
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  P.A.R.T.N.E.R Full System Integration Test                 ║"
echo "║  Duration: 5 minutes                                         ║"
echo "║  Profiling: Enabled on :6060                                 ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Check if binary exists
if [ ! -f "bin/system-integration-test" ]; then
    echo -e "${YELLOW}Building integration test binary...${NC}"
    ASSUME_NO_MOVING_GC_UNSAFE_RISK_IT_WITH=go1.25 go build -o bin/system-integration-test ./cmd/system-integration-test
    echo -e "${GREEN}✓ Build complete${NC}"
fi

# Create reports directory
mkdir -p reports

# Generate timestamp for this test run
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
REPORT_FILE="reports/integration_test_${TIMESTAMP}.json"

echo -e "${BLUE}Starting 5-minute integration test...${NC}"
echo -e "${CYAN}Report will be saved to: ${REPORT_FILE}${NC}"
echo ""
echo -e "${YELLOW}Profiling server available at: http://localhost:6060/debug/pprof/${NC}"
echo -e "${YELLOW}Tip: Run 'go tool pprof http://localhost:6060/debug/pprof/profile' in another terminal${NC}"
echo ""
echo -e "${GREEN}Press Ctrl+C to stop early (will still generate report)${NC}"
echo ""

# Run the test
ASSUME_NO_MOVING_GC_UNSAFE_RISK_IT_WITH=go1.25 \
    ./bin/system-integration-test \
    -duration=5m \
    -profile \
    -export="${REPORT_FILE}" \
    -verbose

echo ""
echo -e "${GREEN}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║  Test Complete!                                              ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${CYAN}Report saved to:${NC} ${REPORT_FILE}"
echo ""

# Generate summary
if [ -f "$REPORT_FILE" ]; then
    echo -e "${BLUE}Quick Summary:${NC}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    # Extract key metrics using jq if available
    if command -v jq &> /dev/null; then
        echo -e "${YELLOW}Duration:${NC}          $(jq -r '.test_info.duration' "$REPORT_FILE")"
        echo -e "${YELLOW}Captures:${NC}          $(jq -r '.metrics.TotalCaptures' "$REPORT_FILE")"
        echo -e "${YELLOW}Capture Success:${NC}   $(jq -r '.metrics.SuccessfulCaptures' "$REPORT_FILE") / $(jq -r '.metrics.TotalCaptures' "$REPORT_FILE")"
        echo -e "${YELLOW}Actual FPS:${NC}        $(jq -r '.metrics.ActualFPS' "$REPORT_FILE" | xargs printf "%.2f")"
        echo -e "${YELLOW}Avg Inference:${NC}     $(echo "scale=2; $(jq -r '.metrics.InferenceTimeSum' "$REPORT_FILE") / $(jq -r '.metrics.TotalInferences' "$REPORT_FILE") / 1000000" | bc)ms"
        echo -e "${YELLOW}Peak Memory:${NC}       $(jq -r '.metrics.PeakMemoryMB' "$REPORT_FILE" | xargs printf "%.2f") MB"
        echo -e "${YELLOW}Avg CPU:${NC}           $(jq -r '.metrics.AvgCPUPercent' "$REPORT_FILE" | xargs printf "%.1f")%%"
        echo -e "${YELLOW}Samples Stored:${NC}    $(jq -r '.metrics.SamplesStored' "$REPORT_FILE")"
    else
        echo -e "${YELLOW}Install 'jq' for detailed metrics parsing${NC}"
        echo "Report location: $REPORT_FILE"
    fi
    
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
fi

echo ""
echo -e "${CYAN}Available commands:${NC}"
echo "  View full report:   cat $REPORT_FILE | jq ."
echo "  Analyze with pprof: go tool pprof -http=:8080 http://localhost:6060/debug/pprof/profile"
echo ""
