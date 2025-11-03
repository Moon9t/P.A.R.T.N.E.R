#!/bin/bash
# Monitor GitHub Actions workflows in real-time
set -e

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}╔═══════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║${NC}  GitHub Actions Workflow Monitor                        ${BLUE}║${NC}"
echo -e "${BLUE}╚═══════════════════════════════════════════════════════════╝${NC}"
echo ""

# Check if gh CLI is installed
if ! command -v gh &> /dev/null; then
    echo -e "${RED}✗ GitHub CLI (gh) not installed${NC}"
    echo ""
    echo "Install with:"
    echo "  Ubuntu/Debian: sudo apt install gh"
    echo "  macOS: brew install gh"
    echo "  Or: https://cli.github.com/"
    exit 1
fi

# Check if authenticated
if ! gh auth status &> /dev/null; then
    echo -e "${YELLOW}⚠ Not authenticated with GitHub${NC}"
    echo ""
    echo "Run: gh auth login"
    exit 1
fi

echo -e "${GREEN}✓ GitHub CLI authenticated${NC}"
echo ""

# Get recent workflow runs
echo -e "${BLUE}Recent Workflow Runs:${NC}"
echo ""
gh run list --limit 10 --json databaseId,displayTitle,conclusion,status,createdAt,workflowName | \
    jq -r '.[] | "\(.workflowName) | \(.displayTitle) | \(.status) | \(.conclusion // "running")"' | \
    while IFS='|' read -r workflow title status conclusion; do
        case "$conclusion" in
            success)
                echo -e "${GREEN}✓${NC} $workflow - $status"
                ;;
            failure)
                echo -e "${RED}✗${NC} $workflow - FAILED"
                ;;
            cancelled)
                echo -e "${YELLOW}⊘${NC} $workflow - Cancelled"
                ;;
            *)
                echo -e "${BLUE}●${NC} $workflow - $status"
                ;;
        esac
    done

echo ""
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

# Watch mode
read -p "Watch for new runs? (y/n): " watch_mode

if [[ "$watch_mode" =~ ^[Yy]$ ]]; then
    echo ""
    echo -e "${YELLOW}Watching for workflow changes... (Ctrl+C to stop)${NC}"
    echo ""
    
    gh run watch
fi
