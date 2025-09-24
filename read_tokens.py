#!/usr/bin/env python3
"""
Read token IDs from binary file created by tokenize_text.py
"""

import sys
import struct

def read_tokens(filename):
    """Read token IDs from binary file."""
    with open(filename, 'rb') as f:
        # Read number of tokens
        count_bytes = f.read(4)
        if len(count_bytes) != 4:
            raise ValueError("Could not read token count")

        count = struct.unpack('<I', count_bytes)[0]
        print(f"Number of tokens: {count}")

        # Read token IDs
        tokens = []
        for i in range(count):
            token_bytes = f.read(4)
            if len(token_bytes) != 4:
                raise ValueError(f"Could not read token {i}")
            token_id = struct.unpack('<I', token_bytes)[0]
            tokens.append(token_id)

        print(f"Token IDs: {tokens}")
        return tokens

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python read_tokens.py <binary_file>")
        sys.exit(1)

    read_tokens(sys.argv[1])