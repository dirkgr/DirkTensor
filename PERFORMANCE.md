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

### 4. Use partial_sort instead of full argsort for top-5
- Result: 9% faster (49.48s vs 54.42s) ✓
- Real: 49.48s, User: 68.89s, Sys: 51.61s
- Sorting 100k+ tokens was expensive; partial_sort much better

### 5. Pre-allocate buffers and use xt::noalias()
- Result: SLOWER (53.09s vs 49.48s)
- Reverted

## Summary

**Progress: 56.92s → 46.25s (19% improvement)**
**Still 2.4x slower than Python (19.05s)**

### Successful optimizations:
1. Enable -O3 for RelWithDebInfo: 7% faster
2. Use partial_sort for top-k: 9% faster
3. (Unexplained improvement after revert: ~3s faster, possibly build artifacts or measurement variance)

### Failed optimizations:
1. Adding xt::eval() everywhere: Made it slower
2. Explicit BLAS linkage: Made it slower (may have been implicit before)
3. Pre-allocated buffers with xt::noalias(): Made it slower

## Analysis

Key observations:
- Very high sys time (~51s out of 49s real) suggests memory allocation bottleneck
- User/real ratio only 1.39x (Python has 6.3x) - poor parallelization
- TBB enabled but not helping much - operations may be too small to parallelize
- xtensor expression templates likely creating many temporaries

Likely bottlenecks:
- Memory allocations in forward pass (exp, reshape, concatenate, etc.)
- xtensor not reusing buffers like PyTorch does
- Limited parallelization of small matrix operations

Next to try:
- Use xt::noalias() for in-place operations
- Pre-allocate buffers and reuse them
- Profile with Instruments to find exact hot spots
- Consider rewriting hot paths with manual loops instead of xtensor

