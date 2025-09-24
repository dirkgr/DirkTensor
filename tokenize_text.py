#!/usr/bin/env python3
"""
Tokenize text using the dolma2 tokenizer and write token IDs to a binary file.

Usage:
    python tokenize_text.py "Hello world, this is a test!" output_tokens.bin
"""

import sys
import struct
from transformers import AutoTokenizer

def tokenize_and_save(text: str, output_file: str):
    """Tokenize text and save token IDs to binary file."""

    # Load the dolma2 tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("allenai/dolma2-tokenizer")

    # Tokenize the text
    print(f"Tokenizing text: '{text}'")
    token_ids = tokenizer.encode(text)

    print(f"Generated {len(token_ids)} tokens: {token_ids}")

    # Write to binary file
    # Format: uint32 for count, followed by uint32 for each token ID
    with open(output_file, 'wb') as f:
        # Write number of tokens
        f.write(struct.pack('<I', len(token_ids)))  # little-endian uint32

        # Write each token ID
        for token_id in token_ids:
            f.write(struct.pack('<I', token_id))  # little-endian uint32

    print(f"Token IDs saved to {output_file}")

def main():
    if len(sys.argv) != 3:
        print("Usage: python tokenize_text.py <text> <output_file>")
        print("Example: python tokenize_text.py \"Hello world!\" tokens.bin")
        sys.exit(1)

    text = sys.argv[1]
    output_file = sys.argv[2]

    try:
        tokenize_and_save(text, output_file)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()