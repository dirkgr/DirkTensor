# Debugger Setup for xtensor Pretty Printing

This project includes LLDB formatters for xtensor types to make debugging easier.

## Files

- `.lldbinit` - LLDB initialization file that loads the formatters (only works if cwd lldbinit is enabled)
- `xtensor_lldb.py` - Python script with xtensor formatters

## CLion Setup

If your `~/.lldbinit` has `settings set target.load-cwd-lldbinit false`, you need to manually configure CLion to load the formatters:

### Method 1: Add to Run/Debug Configuration (Recommended)

1. Go to **Run → Edit Configurations...**
2. Select your debug configuration (e.g., "DirkTensor")
3. In the "Before launch" section, click **+** → **Run External Tool** → **+** (to add a new tool)
   - **Name:** Load LLDB Formatters
   - **Program:** Leave empty
   - **Arguments:** Leave empty
   - **Working directory:** Leave empty
4. Instead, look for the **LLDB Init Commands** or debugger settings in the configuration
5. If there's no direct field, we'll use Method 2

### Method 2: Global LLDB Settings (Applies to all projects)

1. Go to **Preferences → Build, Execution, Deployment → Debugger → LLDB**
2. In the "Startup Commands" or "Init Commands" field, add:
   ```
   command script import /Users/dirkgr/Documents/DirkTensor/xtensor_lldb.py
   ```

### Method 3: Per-Configuration LLDB Commands (If available in your CLion version)

Some CLion versions allow you to add LLDB commands per run configuration:

1. **Run → Edit Configurations...**
2. Select your configuration
3. Look for "LLDB Startup Commands" or similar
4. Add:
   ```
   command script import ${PROJECT_DIR}/xtensor_lldb.py
   ```

## What Gets Pretty Printed

The formatters handle:
- `xt::xtensor<T, N>` - Fixed-rank tensors
- `xt::xarray<T>` - Dynamic-rank tensors
- Expression types - Results from operations like `xt::linalg::tensordot()`, `xt::reshape_view()`, etc. (may show as `const result_type` or `xfunction` types)
- **Summary line**: Shows shape and prefix of formatted values (first ~60 chars)
- **Child "values"**: Full formatted tensor with xtensor-style nested braces
- **Floating point values**: Automatically rounded to 6 significant digits

Example output:
```
simple_tensor = shape=(2, 3) {{1, 2, 3}, {4, 5, ...
  └─ values: {{1, 2, 3}, {4, 5, 6}}

float_tensor = shape=(2, 3) {{0.0120394,  0.165066, -0.0332385}, {0.0480167, -0.125837, ...
  └─ values: {{0.0120394,  0.165066, -0.0332385}, {0.0480167, -0.125837, -0.0140682}}

tensor_3d = shape=(2, 3, 4) {{{1, 1, 1, 1}, {1, ...
  └─ values: {{{1, 1, 1, 1}, {1, 1, 1, 1}, {1, 1, 1, 1}}, {{1, 1, 1, 1}, {1, 1, 1, 1}, {1, 1, 1, 1}}}

projected_qs = shape=(1, 4, 2048) {{{ 0.570252,  0.350372, -0.319389, ...
  └─ values: {{{ 0.570252,  0.350372, -0.319389, ..., 0.0130098, 0.462102, -0.00518558}, {...}}}
```

Note:
- The summary line is a prefix of the full "values" child - same formatting, just truncated.
- Floating point values are rounded to 6 significant digits (e.g., 0.165066123 → 0.165066)

### Viewing in CLion
1. When you hover over a tensor variable, you'll see the summary line
2. **Click the dropdown arrow** to expand the variable
3. Click on the **"values"** child to see the full formatted tensor
4. For large tensors, it shows first 3 and last 3 elements per dimension with `...` in between

## Testing

To verify the formatters are working:

1. Build and run the test: `cmake --build cmake-build-debug --target test_debug_xtensor`
2. Run in LLDB:
   ```bash
   lldb cmake-build-debug/test_debug_xtensor
   (lldb) br set -f test_debug_xtensor.cpp -l 11
   (lldb) run
   (lldb) frame variable
   ```
3. You should see:
   ```
   (xt::xtensor<float, 2>) simple_tensor = shape=(2, 3) {{1, 2, 3}, {4, 5, ...
   (xt::xtensor<float, 3>) tensor_3d = shape=(2, 3, 4) {{{1, 1, 1, 1}, {1, ...
   ```
4. To see the full formatted tensor, use:
   ```
   (lldb) frame variable simple_tensor.values
   (const char[71]) values = "{{1, 2, 3}, {4, 5, 6}}"
   ```

   Note that the summary line is a prefix of the full values.

In CLion:
1. Set a breakpoint in your code where an xtensor variable exists
2. Run the debugger
3. You should see "xtensor pretty printers loaded" in the LLDB console at startup
4. Hover over or inspect an xtensor variable - it should show the shape and size

## Manual Loading During Debug Session

If nothing else works, you can manually load the formatters in the LLDB console during a debug session:

1. Start debugging in CLion (set a breakpoint and run debugger)
2. Open the **Debug → LLDB** console tab
3. Type:
   ```
   command script import /Users/dirkgr/Documents/DirkTensor/xtensor_lldb.py
   ```
4. You should see "xtensor pretty printers loaded"
5. Continue debugging - xtensor variables will now be pretty printed

Note: You'll need to do this each time you start a new debug session unless you configure one of the methods above.
