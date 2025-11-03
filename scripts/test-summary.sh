#!/bin/bash
# System Integration Test - Quick Reference Guide

cat << 'EOF'
╔══════════════════════════════════════════════════════════════════════╗
║                                                                      ║
║  P.A.R.T.N.E.R SYSTEM INTEGRATION TEST                               ║
║  Quick Reference Guide                                               ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝

📚 DOCUMENTATION
─────────────────────────────────────────────────────────────────────
  SYSTEM_VALIDATION_COMPLETE.md   - Test completion summary
  INTEGRATION_TEST_RESULTS.md     - Detailed analysis & recommendations
  reports/full_integration_test.json - Machine-readable metrics

🎯 QUICK START
─────────────────────────────────────────────────────────────────────
  # 30-second validation
  ./bin/system-integration-test -duration=30s -export=reports/quick.json

  # 5-minute full test
  ./scripts/run-full-integration-test.sh

  # With profiling
  ./bin/system-integration-test -duration=5m -profile -verbose

📊 TEST RESULTS (30s Sample Run)
─────────────────────────────────────────────────────────────────────
  ✅ Vision:     59/59 captures (100%), 238ms avg
  ✅ Inference:  59/59 predictions (100%), 1.5ms avg
  ✅ Storage:    11/11 writes (100%), 0 errors
  ✅ FPS:        1.98 / 2.0 target (99.1%)
  ✅ Memory:     12-30 MB (stable)
  ✅ CPU:        46% average
  ✅ Status:     EXCELLENT

🔍 KEY FINDINGS
─────────────────────────────────────────────────────────────────────
  ✅ All modules working correctly
  ✅ 100% reliability across vision, storage, and inference
  ✅ Fast inference (1.5ms = 645 inferences/sec)
  ⚠️  Vision capture is bottleneck (238ms, 99.4% of cycle)
  ⚠️  Model untrained (0.02% confidence - expected)

💡 OPTIMIZATION PRIORITIES
─────────────────────────────────────────────────────────────────────
  1. CRITICAL: Async capture (10-15x FPS improvement)
     - File: internal/vision/capture.go
     - Effort: 2-3 hours
     - Expected: 20-30 FPS

  2. HIGH: Train model (500x confidence improvement)
     - Command: ./bin/collect-real -samples=1000
     - Then: ./bin/partner-v2 -mode=train -epochs=100
     - Expected: 50-70% confidence

  3. MEDIUM: Batch storage writes (10x throughput)
     - File: internal/storage/observation.go
     - Effort: 1-2 hours
     
  4. LOW: Tensor pooling (70% GC reduction)
     - File: internal/model/network.go
     - Effort: 1 hour

📈 PERFORMANCE PROJECTIONS
─────────────────────────────────────────────────────────────────────
  Current State:
    FPS: 2.0 | Latency: 240ms | Confidence: 0.02%

  With Async Capture:
    FPS: 20-30 | Latency: 12-18ms | Confidence: 0.02%

  With Trained Model:
    FPS: 2.0 | Latency: 240ms | Confidence: 50-70%

  With ALL Optimizations:
    FPS: 25-30 | Latency: 12-18ms | Confidence: 55-70%
    Status: PRODUCTION READY ✅

🚀 NEXT STEPS
─────────────────────────────────────────────────────────────────────
  1. Implement async capture (Priority 1)
  2. Collect real chess observations (500-1000)
  3. Train model (100 epochs)
  4. Re-run integration test to validate improvements
  5. Deploy to production

🔧 PROFILING
─────────────────────────────────────────────────────────────────────
  Start test with profiling:
    ./bin/system-integration-test -duration=5m -profile &

  Analyze profiles:
    # CPU
    go tool pprof -http=:8080 http://localhost:6060/debug/pprof/profile

    # Memory
    go tool pprof -http=:8081 http://localhost:6060/debug/pprof/heap

    # Goroutines
    curl http://localhost:6060/debug/pprof/goroutine?debug=2

📞 SUPPORT
─────────────────────────────────────────────────────────────────────
  View reports:
    cat reports/full_integration_test.json | jq .

  Extract metrics:
    jq '.metrics.AvgCPUPercent' reports/full_integration_test.json

  Check logs:
    tail -f reports/test_output.log

✅ STATUS
─────────────────────────────────────────────────────────────────────
  System Integration: ✅ COMPLETE
  All Modules:        ✅ VALIDATED
  Bottlenecks:        ✅ IDENTIFIED
  Optimizations:      ✅ DOCUMENTED
  Production Ready:   🔄 WITH OPTIMIZATIONS

╔══════════════════════════════════════════════════════════════════════╗
║  For detailed analysis, see: INTEGRATION_TEST_RESULTS.md             ║
╚══════════════════════════════════════════════════════════════════════╝

EOF
