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

