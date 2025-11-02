# How to Run P.A.R.T.N.E.R Programs

## The Gorgonia Environment Variable

P.A.R.T.N.E.R uses Gorgonia tensors which require this environment variable for Go 1.25:

```bash
ASSUME_NO_MOVING_GC_UNSAFE_RISK_IT_WITH=go1.25
```

## ✅ Recommended: Use `./run.sh`

The easiest way to run any binary:

```bash
# Show usage
./run.sh

# Test adapter system
./run.sh test-adapter

# Run partner-cli
./run.sh partner-cli --help
./run.sh partner-cli --mode=train --epochs=50

# Run any other binary
./run.sh train-cnn --dataset=data/chess_dataset.db
./run.sh test-model
```

## ✅ Alternative: Use `make`

The Makefile automatically sets the environment variable:

```bash
make test-adapter    # Test adapter system
make build-tools     # Build all tools
make test-model      # Test model
```

## ⚠️ Manual: Set Environment Variable

Only if you must run binaries directly:

```bash
# Export for session
export ASSUME_NO_MOVING_GC_UNSAFE_RISK_IT_WITH=go1.25
./bin/test-adapter

# Or inline
ASSUME_NO_MOVING_GC_UNSAFE_RISK_IT_WITH=go1.25 ./bin/test-adapter
```

## Common Tasks

### Test Adapter System
```bash
./run.sh test-adapter
```

### Build Everything
```bash
make build-tools
```

### Train Model
```bash
./run.sh train-cnn --dataset=data/chess_dataset.db --epochs=100
```

### Get Help
```bash
./run.sh partner-cli --help
make help
```

## Troubleshooting

**Error: panic: Something in this program imports go4.org/unsafe/assume-no-moving-gc...**

→ Use `./run.sh` or `make` instead of running `bin/*` directly!

**Error: Binary not found**

→ Run `make build-tools` first

---

**TL;DR:** Always use `./run.sh <binary>` or `make <target>` 🎯
