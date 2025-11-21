import pytest
import subprocess
import tempfile
import struct
import os
import sys

# Add parent directory to path to import the module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_tokenize_to_file():
    """Test tokenizing text to a file."""
    with tempfile.NamedTemporaryFile(suffix='.bin', delete=False) as f:
        temp_file = f.name

    try:
        # Run tokenization
        result = subprocess.run(
            ['python3', 'tokenize_text.py', 'Hello world', '-o', temp_file],
            capture_output=True,
            text=False
        )

        assert result.returncode == 0, f"Process failed with stderr: {result.stderr}"

        # Read and verify tokens were written
        with open(temp_file, 'rb') as f:
            tokens = []
            while True:
                data = f.read(4)
                if not data:
                    break
                token = struct.unpack('<I', data)[0]
                tokens.append(token)

        # Should have tokenized into at least 1 token
        assert len(tokens) > 0, "No tokens were written"

        # Tokens should be valid uint32 values
        for token in tokens:
            assert 0 <= token <= 0xFFFFFFFF, f"Invalid token value: {token}"

    finally:
        # Clean up
        if os.path.exists(temp_file):
            os.remove(temp_file)


def test_tokenize_to_stdout():
    """Test tokenizing text to stdout."""
    result = subprocess.run(
        ['python3', 'tokenize_text.py', 'Hello world'],
        capture_output=True,
        text=False
    )

    assert result.returncode == 0, f"Process failed with stderr: {result.stderr}"

    # Should have some output
    assert len(result.stdout) > 0, "No data written to stdout"

    # Output length should be multiple of 4 (uint32 size)
    assert len(result.stdout) % 4 == 0, f"Output size {len(result.stdout)} is not a multiple of 4"

    # Parse tokens from stdout
    tokens = []
    for i in range(0, len(result.stdout), 4):
        token = struct.unpack('<I', result.stdout[i:i+4])[0]
        tokens.append(token)

    assert len(tokens) > 0, "No tokens parsed from stdout"


def test_tokenize_empty_string():
    """Test tokenizing an empty string."""
    result = subprocess.run(
        ['python3', 'tokenize_text.py', ''],
        capture_output=True,
        text=False
    )

    # Should still succeed (tokenizers often add special tokens even for empty input)
    assert result.returncode == 0, f"Process failed with stderr: {result.stderr}"


def test_tokenize_special_characters():
    """Test tokenizing text with special characters."""
    test_texts = [
        "Hello, world!",
        "Test with numbers: 123 456",
        "Special chars: @#$%^&*()",
        "Unicode: 你好世界 🌍",
        "Newlines\nand\ttabs"
    ]

    for text in test_texts:
        result = subprocess.run(
            ['python3', 'tokenize_text.py', text],
            capture_output=True,
            text=False
        )

        assert result.returncode == 0, f"Failed to tokenize: {text}"
        assert len(result.stdout) > 0, f"No output for text: {text}"
        assert len(result.stdout) % 4 == 0, f"Invalid output size for text: {text}"


def test_tokenize_long_text():
    """Test tokenizing a longer piece of text."""
    long_text = " ".join(["word" + str(i) for i in range(100)])

    result = subprocess.run(
        ['python3', 'tokenize_text.py', long_text],
        capture_output=True,
        text=False
    )

    assert result.returncode == 0, "Failed to tokenize long text"
    assert len(result.stdout) > 0, "No output for long text"

    # Should produce multiple tokens
    num_tokens = len(result.stdout) // 4
    assert num_tokens > 1, "Long text should produce multiple tokens"


def test_missing_arguments():
    """Test that the script fails gracefully with missing arguments."""
    result = subprocess.run(
        ['python3', 'tokenize_text.py'],
        capture_output=True,
        text=True
    )

    assert result.returncode != 0, "Should fail with no arguments"
    assert "usage:" in result.stderr, "Should show usage message"


def test_file_overwrite():
    """Test that the script can overwrite an existing file."""
    with tempfile.NamedTemporaryFile(suffix='.bin', delete=False) as f:
        temp_file = f.name
        f.write(b'existing data')

    try:
        # First tokenization
        result1 = subprocess.run(
            ['python3', 'tokenize_text.py', 'First text', '-o', temp_file],
            capture_output=True
        )
        assert result1.returncode == 0

        with open(temp_file, 'rb') as f:
            data1 = f.read()

        # Second tokenization (should overwrite)
        result2 = subprocess.run(
            ['python3', 'tokenize_text.py', 'Different text', '-o', temp_file],
            capture_output=True
        )
        assert result2.returncode == 0

        with open(temp_file, 'rb') as f:
            data2 = f.read()

        # Data should be different
        assert data1 != data2, "File was not overwritten"

    finally:
        if os.path.exists(temp_file):
            os.remove(temp_file)


def test_tokenize_from_file():
    """Test tokenizing text from an input file."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as input_f:
        input_file = input_f.name
        input_f.write("Hello from file!")

    with tempfile.NamedTemporaryFile(suffix='.bin', delete=False) as output_f:
        output_file = output_f.name

    try:
        # Run tokenization from file
        result = subprocess.run(
            ['python3', 'tokenize_text.py', '--file', input_file, '-o', output_file],
            capture_output=True,
            text=False
        )

        assert result.returncode == 0, f"Process failed with stderr: {result.stderr}"

        # Read and verify tokens were written
        with open(output_file, 'rb') as f:
            tokens = []
            while True:
                data = f.read(4)
                if not data:
                    break
                token = struct.unpack('<I', data)[0]
                tokens.append(token)

        # Should have tokenized into at least 1 token
        assert len(tokens) > 0, "No tokens were written"

    finally:
        # Clean up
        if os.path.exists(input_file):
            os.remove(input_file)
        if os.path.exists(output_file):
            os.remove(output_file)


def test_tokenize_from_file_to_stdout():
    """Test tokenizing text from a file to stdout."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as input_f:
        input_file = input_f.name
        input_f.write("Hello from file to stdout!")

    try:
        result = subprocess.run(
            ['python3', 'tokenize_text.py', '-f', input_file],
            capture_output=True,
            text=False
        )

        assert result.returncode == 0, f"Process failed with stderr: {result.stderr}"

        # Should have some output
        assert len(result.stdout) > 0, "No data written to stdout"

        # Output length should be multiple of 4 (uint32 size)
        assert len(result.stdout) % 4 == 0, f"Output size {len(result.stdout)} is not a multiple of 4"

    finally:
        if os.path.exists(input_file):
            os.remove(input_file)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])