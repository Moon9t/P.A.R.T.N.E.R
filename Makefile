.PHONY: all build clean install test run-advise run-train run-watch deps

# Binary name
BINARY=partner
BUILD_DIR=bin
CMD_DIR=cmd/partner

# Build flags
LDFLAGS=-ldflags="-s -w"
GOFLAGS=-trimpath

all: clean deps build

# Build the application
build:
	@echo "Building P.A.R.T.N.E.R..."
	@mkdir -p $(BUILD_DIR)
	@go build $(GOFLAGS) $(LDFLAGS) -o $(BINARY) $(CMD_DIR)/main.go
	@echo "Build complete: ./$(BINARY)"

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

# Show help
help:
	@echo "P.A.R.T.N.E.R Makefile"
	@echo ""
	@echo "Usage:"
	@echo "  make build        - Build the application"
	@echo "  make deps         - Install dependencies"
	@echo "  make setup        - Create required directories"
	@echo "  make clean        - Clean build artifacts"
	@echo "  make run-advise   - Run in advise mode"
	@echo "  make run-train    - Run in train mode"
	@echo "  make run-watch    - Run in watch mode"
	@echo "  make install      - Install to /usr/local/bin"
	@echo "  make uninstall    - Uninstall from /usr/local/bin"
	@echo "  make fmt          - Format code"
	@echo "  make test         - Run tests"
	@echo "  make help         - Show this help"
