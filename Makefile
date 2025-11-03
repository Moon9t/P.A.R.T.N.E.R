.PHONY: all build clean install test run-advise run-train run-watch deps

# Build directory
BUILD_DIR=bin

# Build flags
LDFLAGS=-ldflags="-s -w"
GOFLAGS=-trimpath

all: clean deps build

# Build all binaries
build:
	@echo "Building P.A.R.T.N.E.R tools..."
	@mkdir -p $(BUILD_DIR)
	@echo "  train-cnn..."
	@go build $(GOFLAGS) $(LDFLAGS) -o $(BUILD_DIR)/train-cnn cmd/train-cnn/main.go
	@echo "  ingest-pgn..."
	@go build $(GOFLAGS) $(LDFLAGS) -o $(BUILD_DIR)/ingest-pgn cmd/ingest-pgn/main.go
	@echo "  live-chess..."
	@go build $(GOFLAGS) $(LDFLAGS) -o $(BUILD_DIR)/live-chess cmd/live-chess/main.go
	@echo "  live-analysis..."
	@go build $(GOFLAGS) $(LDFLAGS) -o $(BUILD_DIR)/live-analysis cmd/live-analysis/main.go
	@echo "✓ Build complete"

# Install dependencies
deps:
	@echo "Installing dependencies..."
	@go mod download
	@go mod tidy
	@echo "Dependencies installed"

# Create required directories
setup:
	@echo "Setting up directories..."
	@mkdir -p data logs
	@echo "Setup complete"

# Clean build artifacts
clean:
	@echo "Cleaning..."
	@rm -f $(BINARY)
	@rm -rf $(BUILD_DIR)
	@echo "Clean complete"

# Run tests
test:
	@echo "Running tests..."
	@go test -v ./...

# Test model implementation
test-model: setup
	@echo "Testing ML model implementation..."
	@go run cmd/test-model/main.go

# Build all tools
build-tools:
	@echo "Building all tools..."
	@mkdir -p $(BUILD_DIR)
	@ASSUME_NO_MOVING_GC_UNSAFE_RISK_IT_WITH=go1.25 go build -o $(BUILD_DIR)/ingest-pgn ./cmd/ingest-pgn
	@ASSUME_NO_MOVING_GC_UNSAFE_RISK_IT_WITH=go1.25 go build -o $(BUILD_DIR)/train-cnn ./cmd/train-cnn
	@ASSUME_NO_MOVING_GC_UNSAFE_RISK_IT_WITH=go1.25 go build -o $(BUILD_DIR)/test-model ./cmd/test-model
	@ASSUME_NO_MOVING_GC_UNSAFE_RISK_IT_WITH=go1.25 go build -o $(BUILD_DIR)/self-improvement ./cmd/self-improvement-demo
	@ASSUME_NO_MOVING_GC_UNSAFE_RISK_IT_WITH=go1.25 go build -o $(BUILD_DIR)/live-chess ./cmd/live-chess
	@echo "All tools built successfully!"

# Run self-improvement system
run-self-improve:
	@echo "Running self-improvement system..."
	@mkdir -p $(BUILD_DIR) data/models data/replays
	@ASSUME_NO_MOVING_GC_UNSAFE_RISK_IT_WITH=go1.25 ./bin/self-improvement \
		--model data/models/chess_cnn.bin \
		--dataset data/positions.db \
		--observations 100

# Run live chess vision analysis
run-live-chess:
	@echo "Running live chess vision analysis..."
	@mkdir -p $(BUILD_DIR) data/models
	@ASSUME_NO_MOVING_GC_UNSAFE_RISK_IT_WITH=go1.25 ./bin/live-chess \
		--model data/models/chess_cnn.bin \
		--x 100 --y 100 --width 800 --height 800 \
		--fps 2 --top 5

# Test adapter system
test-adapter:
	@echo "Testing Game Adapter Interface..."
	@mkdir -p $(BUILD_DIR)
	@ASSUME_NO_MOVING_GC_UNSAFE_RISK_IT_WITH=go1.25 go build -o $(BUILD_DIR)/test-adapter cmd/test-adapter/main.go
	@ASSUME_NO_MOVING_GC_UNSAFE_RISK_IT_WITH=go1.25 $(BUILD_DIR)/test-adapter
	@echo ""

# Run in advise mode
run-advise: build setup
	@echo "Starting in ADVISE mode..."
	@./$(BINARY) -mode=advise

# Run in train mode
run-train: build setup
	@echo "Starting in TRAIN mode..."
	@./$(BINARY) -mode=train -epochs=100

# Run in watch mode
run-watch: build setup
	@echo "Starting in WATCH mode..."
	@./$(BINARY) -mode=watch

# Install to system
install: build
	@echo "Installing to /usr/local/bin..."
	@sudo cp $(BINARY) /usr/local/bin/
	@echo "Installed successfully"

# Uninstall from system
uninstall:
	@echo "Uninstalling from /usr/local/bin..."
	@sudo rm -f /usr/local/bin/$(BINARY)
	@echo "Uninstalled successfully"

# Run with custom config
run-custom: build setup
	@./$(BINARY) -config=$(CONFIG) -mode=$(MODE)

# Format code
fmt:
	@echo "Formatting code..."
	@go fmt ./...
	@echo "Format complete"

# Lint code (requires golangci-lint)
lint:
	@echo "Linting code..."
	@golangci-lint run
	@echo "Lint complete"

# Run quick start guide
quick-start:
	@./scripts/03-interactive-setup.sh

# Run full workflow (requires PGN file)
workflow:
	@./scripts/05-complete-pipeline.sh

# Run integration tests
test-integration:
	@echo "Running integration tests..."
	@./scripts/06-integration-tests.sh

# Check system status
status:
	@./scripts/02-system-status.sh

# Run quick demo
demo:
	@./scripts/01-quick-demo.sh

# Show next steps guide
next:
	@./scripts/04-production-ready.sh

# Show help
help:
	@echo "P.A.R.T.N.E.R Makefile"
	@echo ""
	@echo "Quick Start:"
	@echo "  make status            - Check system status and readiness"
	@echo "  make demo              - Run quick 60-second demo"
	@echo "  make next              - Production setup guide (recommended!)"
	@echo "  make quick-start       - Interactive quick start guide"
	@echo "  make workflow          - Run complete workflow (PGN → Train → Improve)"
	@echo "  make test-integration  - Run integration tests"
	@echo ""
	@echo "Build Commands:"
	@echo "  make build             - Build the application"
	@echo "  make build-tools       - Build all tools (ingest-pgn, train-cnn, etc.)"
	@echo "  make clean             - Clean build artifacts"
	@echo "  make deps              - Install dependencies"
	@echo "  make setup             - Create required directories"
	@echo ""
	@echo "Run Commands:"
	@echo "  make run-live-chess    - Run live chess vision analysis"
	@echo "  make run-self-improve  - Run self-improvement system"
	@echo ""
	@echo "Development:"
	@echo "  make fmt               - Format code"
	@echo "  make lint              - Lint code"
	@echo "  make test              - Run tests"
	@echo ""
	@echo "Installation:"
	@echo "  make install           - Install to /usr/local/bin"
	@echo "  make uninstall         - Uninstall from /usr/local/bin"
