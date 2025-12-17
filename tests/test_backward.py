import pytest
import subprocess
import os
import sys
import time

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run_cpp(token_file):
    """Run C++ implementation and return (loss_before, loss_after, elapsed_seconds)."""
    start = time.time()
    result = subprocess.run(
        ['cmake-build-relwithdebinfo/DirkTensor', token_file],
        capture_output=True,
        text=True,
        timeout=120
    )
    elapsed = time.time() - start
    assert result.returncode == 0, f"C++ failed: {result.stderr}"
    lines = result.stdout.strip().split('\n')
    return float(lines[0]), float(lines[1]), elapsed


def run_python(token_file):
    """Run Python implementation and return (loss_before, loss_after, elapsed_seconds)."""
    start = time.time()
    result = subprocess.run(
        ['python3', 'backward.py', token_file],
        capture_output=True,
        text=True,
        timeout=180
    )
    elapsed = time.time() - start
    assert result.returncode == 0, f"Python failed: {result.stderr}"
    lines = result.stdout.strip().split('\n')
    return float(lines[0]), float(lines[1]), elapsed


@pytest.mark.parametrize("token_file", [
    "fourscore.tokens.bin",
    "data/benchmark64.tokens.bin",
])
def test_backward_matches_python(token_file):
    """Test that C++ and Python produce matching loss values, and C++ is faster."""
    if not os.path.exists(token_file):
        pytest.skip(f"Token file {token_file} not found")

    cpp_before, cpp_after, cpp_time = run_cpp(token_file)
    py_before, py_after, py_time = run_python(token_file)

    # Print comparison table
    print(f"\n{'':10} {'Before':>15} {'After':>15} {'Time (s)':>15}")
    print(f"{'C++':10} {cpp_before:>15.5f} {cpp_after:>15.5f} {cpp_time:>15.2f}")
    print(f"{'Python':10} {py_before:>15.5f} {py_after:>15.5f} {py_time:>15.2f}")
    print(f"{'Speedup':10} {'':>15} {'':>15} {py_time/cpp_time:>14.1f}x")

    # Check values are close (relative tolerance of 1e-3)
    assert abs(cpp_before - py_before) / py_before < 1e-3, \
        f"Loss before mismatch: C++={cpp_before}, Python={py_before}"
    assert abs(cpp_after - py_after) / py_after < 1e-3, \
        f"Loss after mismatch: C++={cpp_after}, Python={py_after}"

    # Check that loss decreased (training is working)
    assert cpp_after < cpp_before, "C++ loss should decrease after step"
    assert py_after < py_before, "Python loss should decrease after step"

    # Check that C++ is faster than Python
    assert cpp_time < py_time, \
        f"C++ ({cpp_time:.2f}s) should be faster than Python ({py_time:.2f}s)"


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
