# Performance Benchmarks

## Baseline (VibeSpeed branch start)

Command: `time <program> fourscore.tokens.bin`

### Python inference.py
- Real: 19.05s
- User: 119.75s (highly parallelized)
- Sys: 13.35s

### C++ DirkTensor
- Real: 56.92s
- User: 69.07s
- Sys: 55.49s

**C++ is 3x slower than Python! High sys time suggests I/O or memory issues.**

## Optimization Attempts

### 1. Add xt::eval() calls everywhere
- Result: SLOWER (60.21s vs 56.92s)
- Reverted

### 2. Enable -O3 for RelWithDebInfo
- Result: 7% faster (51.09s vs 54.74s) ✓
- Real: 51.09s, User: 69.89s, Sys: 51.42s

### 3. Link to BLAS/LAPACK (Accelerate framework)
- Result: SLOWER (54.42s vs 51.09s)
- Real: 54.42s, User: 73.94s, Sys: 51.20s
- Note: May have already been using BLAS implicitly

## Analysis

**Current best: 51.09s (still 2.7x slower than Python's 19.05s)**

Key observations:
- Very high sys time (~51s) suggests memory allocation or system call bottleneck
- Python's high user time (119.75s) vs low real time (19.05s) shows good parallelization
- C++ not parallelizing well despite TBB being enabled

Next to try:
- Profile to find hot spots
- Reduce memory allocations in attention mechanism
- Check if TBB is actually parallelizing operations

