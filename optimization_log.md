# C++ Performance Optimization Log

## Goal
Optimize the C++ implementation to match or exceed the Python/PyTorch implementation performance.

## System Information
- **Date Started**: 2025-11-19
- **Platform**: macOS Darwin 24.6.0
- **Compiler**: Clang (Apple)
- **Build Type**: Release with -O3 -march=native -ffast-math
- **Libraries**: xtensor 0.27.0, xsimd 13.0.0, xtensor-blas 0.23.0, TBB 2021.13.0, Apple Accelerate
- **Test Files**:
  - archilles.tokens.bin (468 bytes, ~117 tokens)
  - dodonaea.tokens.bin (308 bytes, ~77 tokens)

## Baseline Performance

### Initial Measurements (Before Optimization)
**Date**: 2025-11-19
**Commit**: 8c6bd8f "Add comprehensive timing instrumentation"

#### Python Implementation (inference.py)
- **Command**: `python inference.py data/archilles.tokens.bin`
- **Time**: 2.08 seconds average (5 runs)
- **Tokens/sec**: 56.3
- **Implementation**: PyTorch with HuggingFace Transformers
- **Details**:
  - Model loading: 1.247s
  - Forward pass avg: 2.0765s (min: 1.7684s, max: 2.3868s)

#### C++ Implementation (DirkTensor)
- **Command**: `cmake-build-release/DirkTensor data/archilles.tokens.bin`
- **Time**: 40.25 seconds average per forward pass
- **Tokens/sec**: 2.9
- **Performance vs Python**: **19.4x SLOWER**
- **Details**:
  - Model loading: 7.496s (6x slower than Python)
  - Warm-up pass: 40.252s
  - Run 1: 40.261s
  - Run 2: 40.242s
  - (Stopped early - clear performance issue identified)

---

## Optimization Attempts

### Phase 1: Setup & Baseline

#### Experiment 1.1: Add Timing Infrastructure
**Date**: 2025-11-19
**Hypothesis**: Need accurate timing to measure optimization impact
**Changes**: Added std::chrono timing to C++ and time.perf_counter to Python
**Status**: COMPLETED
**Result**: Successfully added comprehensive timing to both implementations

#### Experiment 1.2: Fix Variable Shadowing
**Date**: 2025-11-19
**Commit**: f91d1c0
**Hypothesis**: Variable shadowing causes confusion and wastes memory
**Changes**: Renamed local max_seq_len to actual_max_len, allocate batch with actual size
**Result**: Cleaner code, ~35x less memory for small inputs
**Performance Impact**: Minimal (main bottleneck is elsewhere)

### Phase 2: High-Impact Optimizations

#### Experiment 2.1: Rewrite Attention Mechanism (ATTEMPTED)
**Date**: 2025-11-19
**Hypothesis**: Nested position-by-position loops are the main bottleneck
**Changes**: Attempted to use xt::linalg::dot for batched GEMM operations
**Status**: REVERTED - Layout issues with xtensor-blas on strided views
**Lesson**: Need simpler optimizations first or different approach to BLAS

#### Experiment 2.2: Vectorize MLP SiLU Activation
**Date**: 2025-11-19
**Hypothesis**: Scalar loop for SiLU can be vectorized with xtensor
**Changes**: Replaced element-wise loop with xtensor expression templates
**Before**: for(i) { x[i] = x[i]/(1+exp(-x[i])) * proj[i] }
**After**: silu = gate / (1 + xt::exp(-gate)) * projected
**Result**: ~5% improvement (38.3s vs 40.25s baseline)
**Status**: COMPLETED - Small improvement

#### Experiment 2.3: Profiling Analysis
**Date**: 2025-11-19
**Method**: Added detailed timing to each component
**Results**: SURPRISING FINDINGS!
- **MLP: 73.1% of time** (45.4s total, ~2.8s per layer)
- **Attention: 26.6% of time** (16.5s total, ~1.0s per layer)
- **Norms: 0.3% of time** (negligible)

**Key Insight**: MLP is the main bottleneck, NOT attention!
- Despite vectorizing SiLU, MLP still takes 2.8s per layer
- Likely culprit: tensordot operations (3 per MLP layer)
- Each tensordot is probably not using optimized BLAS

---

## Performance Tracking Summary

| Phase | Optimization | Before (ms) | After (ms) | Speedup | Status |
|-------|-------------|------------|------------|---------|---------|
| Baseline | Initial | TBD | - | - | Pending |

---

## Key Findings

1. **Initial Analysis**:
   - Attention mechanism uses inefficient nested loops (O(n²))
   - RoPE implementation uses scalar loops instead of vectorization
   - MLP activation processes elements one at a time
   - Recent "optimization" commits actually made performance worse

2. **Reverted Optimizations to Revisit**:
   - Commit 09eb386: Reverted parallelism (was 10 seconds faster)
   - Commit 71a1151: "Big Speedup in rope" actually made it slower
   - Commit 196de11: Removed vectorized SiLU

---

#### Experiment 2.4: Optimize MLP with Direct BLAS Operations
**Date**: 2025-11-19
**Hypothesis**: tensordot is inefficient for batched matrix multiplications
**Changes**:
- Replaced tensordot with reshape + dot operations
- Reshape 3D input to 2D for efficient GEMM
- Use xt::linalg::dot with transposed weights
**Result**: **MASSIVE SUCCESS!**
- MLP time: 45.4s → 1.65s (27.5x speedup!)
- Overall: 40.25s → 19.7s (2x speedup)
- Throughput: 2.9 → 5.9 tokens/sec
**New bottleneck**: Attention now 91.6% of runtime

#### Experiment 2.5: Optimize Attention Projections
**Date**: 2025-11-19
**Hypothesis**: Attention projections (Q,K,V,O) can use same optimization as MLP
**Changes**:
- Applied reshape + dot optimization to Q, K, V projections
- Applied same optimization to output projection
- Kept original attention computation loop (for stability)
**Result**: **SUCCESS!**
- Attention time: 20.9s → 8.2s (2.5x speedup)
- Overall: 19.7s → 13.3s (1.5x speedup)
- Total improvement from baseline: 3x (40.25s → 13.3s)
- Throughput: 5.9 → 8.8 tokens/sec

#### Experiment 2.6: Re-enable TBB Parallelization
**Date**: 2025-11-20
**Hypothesis**: TBB parallelization can improve attention performance
**Changes**:
- Re-applied TBB parallel_for with blocked_range2d
- Parallelized across batch and sequence dimensions
**Result**: **SUCCESS!**
- Overall: 13.3s → 10.2s (1.3x speedup)
- Total improvement from baseline: 3.9x (40.25s → 10.2s)
- Throughput: 8.8 → 11.4 tokens/sec
- Gap to Python: 5.4x slower (improved from 7x)

---

## Next Steps

1. ✅ Create this optimization log
2. ✅ Add timing instrumentation
3. ✅ Measure baseline performance
4. ✅ Fix variable shadowing bug
5. ✅ Optimize MLP (HUGE WIN!)
6. ⏳ Optimize attention mechanism (now main bottleneck)
7. ⏳ Vectorize RoPE
8. ⏳ Reduce materializations
9. ⏳ Re-enable parallelization
---

## Performance Summary

### Final Results (2025-11-20)

**Python Baseline:**
- Forward pass: 1.89 seconds
- Throughput: 61.9 tokens/sec

**C++ Optimized:**
- Forward pass: 10.2 seconds (from 40.25s baseline)
- Throughput: 11.4 tokens/sec (from 2.9 baseline)
- **Total speedup: 3.9x**
- **Gap to Python: 5.4x slower** (improved from 19.4x slower)

### Optimization Impact

| Optimization | Time Reduction | Speedup |
|--------------|---------------|---------|
| MLP with BLAS | 40.25s → 19.7s | 2.0x |
| Attention projections | 19.7s → 13.3s | 1.5x |
| TBB parallelization | 13.3s → 10.2s | 1.3x |
| **Total** | **40.25s → 10.2s** | **3.9x** |

### Current Bottlenecks
- Attention computation loop: 84.7% of runtime (8.2s)
- MLP: 13.5% of runtime (1.3s)
- Norms: 1.8% of runtime (0.2s)

### Phase 3: Further Attention Optimizations

#### Experiment 3.1: Tiled Attention Implementation
**Date**: 2025-11-20
**Hypothesis**: Cache-optimized tiled attention with online softmax might improve performance
**Changes**: Implemented tiled attention with Q_TILE=16, KV_TILE=32 and online softmax
**Result**: **FAILED** - Timed out after 120 seconds (>12x slower!)
**Lesson**: Manual loops and complex tiling logic performed poorly on CPU

#### Experiment 3.2: Vectorized RoPE
**Date**: 2025-11-20
**Hypothesis**: Triple-nested loop in RoPE is inefficient
**Changes**: Process all heads at once instead of individual loops
**Result**: **Minor success** - 10.2s → 10.0s (2% improvement)
**Performance**: 11.7 tokens/sec (from 11.4)

#### Experiment 3.3: Batch Attention Computation
**Date**: 2025-11-20
**Hypothesis**: Process all attention positions at once instead of sequentially
**Changes**: Implemented batch Q@K^T computation and row-wise softmax
**Result**: **FAILED** - "No valid layout chosen" error from xtensor-blas
**Lesson**: Non-contiguous tensor views incompatible with xtensor-blas operations

### Final Performance Summary (2025-11-20)

**Best C++ Performance Achieved:**
- Forward pass: 10.0 seconds
- Throughput: 11.7 tokens/sec
- Total speedup from baseline: 4x (40.25s → 10.0s)
- Gap to Python: 5.3x slower (Python: 61.9 tokens/sec)

**Successful Optimizations:**
1. MLP with direct BLAS: 27.5x speedup for MLP component
2. Attention projections with BLAS: 2.5x speedup for attention
3. TBB parallelization: 1.3x overall speedup
4. Vectorized RoPE: 2% improvement

**Failed Attempts:**
1. Tiled attention with online softmax: >12x slower than baseline
2. Batch attention computation: Layout errors with xtensor-blas

### Key Learnings

1. **xtensor-blas limitations**: Requires contiguous memory layouts, struggles with strided views
2. **BLAS is crucial**: Direct BLAS operations via reshape+dot pattern provide massive speedups
3. **Sequential bottleneck**: Position-by-position attention loop remains main bottleneck (50% of runtime)
4. **PyTorch approach**: Relies on optimized BLAS libraries (bmm) rather than manual optimization

### Remaining Opportunities
1. **Alternative tensor library**: Consider libraries with better BLAS integration for non-contiguous views
2. **Custom SIMD implementation**: Hand-written vectorized attention kernel
3. **Memory layout optimization**: Ensure all tensors are contiguous before operations
4. **Compiler optimizations**: Profile-guided optimization, link-time optimization
