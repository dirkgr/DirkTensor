import pytest
import subprocess
import tempfile
import os
import sys
import shutil
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def temp_output_dir():
    """Create a temporary directory for test outputs."""
    temp_dir = tempfile.mkdtemp(prefix='test_checkpoint_')
    yield temp_dir
    # Cleanup
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)


def test_missing_arguments():
    """Test that the script fails gracefully with missing arguments."""
    result = subprocess.run(
        ['python3', 'convert_checkpoint_from_huggingface.py'],
        capture_output=True,
        text=True
    )

    assert result.returncode != 0, "Should fail with no arguments"
    assert "usage:" in result.stderr.lower(), "Should show usage message"


def test_invalid_model_name(temp_output_dir):
    """Test handling of invalid model names."""
    result = subprocess.run(
        ['python3', 'convert_checkpoint_from_huggingface.py',
         'invalid/model/name/that/does/not/exist', temp_output_dir],
        capture_output=True,
        text=True
    )

    # Should fail when trying to load non-existent model
    assert result.returncode != 0, "Should fail with invalid model name"


def test_small_model_conversion(temp_output_dir):
    """Test with a real small model."""
    # Use a very small model for testing
    model_name = "hf-internal-testing/tiny-random-gpt2"

    result = subprocess.run(
        ['python3', 'convert_checkpoint_from_huggingface.py',
         model_name, temp_output_dir],
        capture_output=True,
        text=True,
        timeout=60  # 1 minute timeout
    )

    assert result.returncode == 0, f"Conversion failed with stderr: {result.stderr}"

    # Check that .npy files were created
    npy_files = [f for f in os.listdir(temp_output_dir) if f.endswith('.npy')]
    assert len(npy_files) > 0, "Should create at least one .npy file"

    # Verify each .npy file is readable and valid
    for npy_file in npy_files:
        filepath = os.path.join(temp_output_dir, npy_file)

        # Check file exists and has size > 0
        assert os.path.exists(filepath), f"File should exist: {npy_file}"
        assert os.path.getsize(filepath) > 0, f"File should not be empty: {npy_file}"

        # Try to load the array
        try:
            array = np.load(filepath)
        except Exception as e:
            pytest.fail(f"Failed to load .npy file {npy_file}: {e}")

        # Verify array properties
        assert array.dtype == np.float32, f"Array should be float32, got {array.dtype} for {npy_file}"
        assert array.size > 0, f"Array should not be empty: {npy_file}"
        assert array.ndim >= 1, f"Array should have at least 1 dimension: {npy_file}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])