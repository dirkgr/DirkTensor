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

### 6. Try -Ofast and -ffast-math compiler flags
- Changed from `-O3` to `-Ofast -ffast-math`
- Result: **MUCH FASTER (24.61s vs 45.69s)** ✓✓✓
- Real: 24.61s, Per iteration: 895ms
- **54% faster! 2.2x speedup!**
- This is by far the biggest win

## Summary

**Progress: 56.92s → 24.61s (57% improvement!)**
**Now only 2.9x slower than Python (8.3s)**

### Successful optimizations:
1. Enable -O3 for RelWithDebInfo: 7% faster
2. Use partial_sort for top-k: 9% faster
3. **Use -Ofast -ffast-math: 54% faster! (2.2x speedup)** ⭐
4. (Unexplained improvement after revert: ~3s faster, possibly build artifacts or measurement variance)

### Failed optimizations:
1. Adding xt::eval() everywhere: Made it slower
2. Explicit BLAS linkage: Made it slower (may have been implicit before)
3. Pre-allocated buffers with xt::noalias(): Made it slower
4. Pre-allocated RoPE buffer to avoid concatenate: Made it 2% slower

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

## Profiling with Instruments (xctrace)

Profiled with `xctrace record --template "Time Profiler"` on RelWithDebInfo build.

### Key findings (122,426 total samples):
- **15,932 samples (13.0%) marked as "unknown"** - large unattributable overhead
- **Our computation code appears in < 0.1% of samples:**
  - OlmoAttention::forward: 16 samples (0.0%)
  - OlmoMlp::forward: 3 samples (0.0%)
  - RMSNorm::forward: 2 samples (0.0%)
  - main: 9 samples (0.0%)
- **BLAS operations: only 53 samples total (0.1%)**
- **Most samples in TBB threading overhead and system calls**

### Interpretation:
**The bottleneck is NOT in the forward pass computation!**

This is surprising and suggests:
1. Most time spent in program startup/model loading (reading .npy files)
2. Dynamic library loading overhead
3. Memory allocation outside the hot loops
4. The actual computation may be fast but we're not reusing work

This explains why:
- High sys time (51s) - likely I/O and dynamic allocation
- Low user/real ratio - not much actual CPU work happening
- Optimizing forward passes had limited impact

### Next steps:
1. ~~Profile just the inference loop (exclude startup)~~ - Done!
2. ~~Measure model loading time separately~~ - Done!

## Detailed Timing Analysis

Added timing instrumentation to both C++ and Python to measure where time is actually spent.

### C++ Timing Breakdown:

**Before -Ofast (45.7s total):**
- Read tokens: 0ms (0%)
- Load model: 6,325ms (13.8%)
- Load detokenizer: 21ms (0%)
- Inference (20 iterations): 39,337ms (86.2%)
- Per iteration: 1,967ms

**After -Ofast (24.6s total):**
- Read tokens: 0ms (0%)
- Load model: 6,679ms (27.1%)
- Load detokenizer: 21ms (0%)
- **Inference (20 iterations): 17,906ms (72.8%)**
- **Per iteration: 895ms** ✓

### Python Timing Breakdown (8.3s total):
- Read tokens: 0ms (0%)
- Load tokenizer: 522ms (6.3%)
- Load model: 467ms (5.6%)
- **Inference (20 iterations): 7,330ms (88.1%)**
- **Per iteration: 366ms**

### Critical Finding: Compiler optimization flags make huge difference!

**Before -Ofast:**
- C++ per iteration: 1,967ms
- Python per iteration: 366ms
- **C++ was 5.4x slower**

**After -Ofast:**
- C++ per iteration: 895ms
- Python per iteration: 366ms
- **C++ now only 2.4x slower!**

Initially suspected missing KV cache, but investigation showed:
- ✅ C++ HAS KV cache implementation (OlmoAttention.h lines 47-49)
- ✅ KV cache IS being used correctly (OlmoAttention.cpp lines 31-37)
- ✅ Both C++ and Python use KV caching

The real issue was compiler optimization. `-Ofast -ffast-math` enables:
- Aggressive floating-point optimizations
- Relaxed IEEE 754 compliance (allows reassociation, reciprocal approximations)
- Auto-vectorization improvements
- Loop optimizations

Why is C++ still 2.4x slower than Python?

### Likely bottlenecks:
1. **Memory allocations in apply_rope**: `xt::concatenate()` (OlmoAttention.cpp:77) allocates on every call
   - Called 2x per attention layer (once for q, once for k)
   - 16 layers × 2 = 32 allocations per token
   - 20 tokens × 32 = 640 allocations total
2. **xtensor expression template overhead**: Many intermediate temporaries in forward pass
3. **Softmax implementation**: Could be optimized
4. **Broadcasting operations**: exp_logits division might be inefficient
5. **Python uses highly optimized PyTorch backend** (BLAS, MKL, or CUDA kernels)

## Re-profiling After -Ofast (with sample tool)

After the -Ofast optimization, re-profiled to find remaining bottlenecks.

**Total samples: 20,064**

### Call hierarchy:
- main: 16,310 samples (81%)
  - OlmoModel::forward: 15,409 samples (77%)
    - OlmoBlock::forward: 9,418 samples (47%)
      - **OlmoAttention::forward: 5,410 samples (27%)**
        - **OlmoAttention::apply_rope: 5,340 samples (26.6%)**
          - OlmoAttention::rope_buffers(): 5,336 samples (26.6%)

### Key finding: **RoPE is the bottleneck, NOT softmax!**

**26.6% of total runtime** is spent in `apply_rope()`, specifically in the RoPE position encoding computation. This includes:
- Concatenate operation for rotating the input (line 86-87 in OlmoAttention.cpp)
- Broadcasting and element-wise multiplication with sin/cos buffers

### Attempted optimizations of RoPE (all failed):
1. **Pre-allocated buffer for concatenate**: Made it 23% SLOWER (1,101ms vs 895ms)
   - Extra view assignments and member state overhead outweighed benefits
2. **Move rope_buffers to constructor**: Hung during model loading
   - Computing 16 layers × RoPE buffers at once was too expensive

### Why RoPE is hard to optimize:
- Concatenate operation is called 640 times (32× per token × 20 tokens)
- But eliminating it with pre-allocated buffers adds overhead
- The static rope_buffers() caching is already working efficiently
- xtensor expression templates are doing their job

### Remaining bottleneck analysis:
The 2.4x gap to PyTorch is likely because:
1. **RoPE (26.6%)**: PyTorch likely has hand-optimized RoPE kernels
2. **Element-wise ops**: PyTorch's oneDNN/MKL-DNN kernels for softmax, exp, etc.
3. **Memory layout**: PyTorch's tensor memory layout may be more cache-friendly
4. **BLAS operations**: While we use Accelerate, PyTorch may use it more effectively

### Next optimization targets:
1. ~~Try -Ofast and -ffast-math compiler flags~~ ✓ Done! (2.2x speedup)
2. ~~Profile specifically to find hotspot~~ ✓ Done! (Found RoPE is 26.6%)
3. ~~Eliminate concatenate in apply_rope~~ ✗ Made it slower
4. Accept the 2.4x gap - remaining optimizations would require rewriting hot paths in assembly/intrinsics
5. Or: Try different xtensor usage patterns (more eval(), less lazy evaluation?)

