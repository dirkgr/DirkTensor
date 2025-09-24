#!/usr/bin/env python3
"""
Tokenize text using the dolma2 tokenizer and write token IDs to a binary file.

Usage:
    python tokenize_text.py "Hello world, this is a test!" output_tokens.bin
"""

import sys
import struct
import logging
from transformers import AutoTokenizer

# Configure logging to stderr (bash-friendly)
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s',
    stream=sys.stderr
)

def tokenize_and_save(text: str, output_file: str):
    """Tokenize text and save token IDs to binary file."""

    # Load the dolma2 tokenizer
    logging.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("allenai/dolma2-tokenizer")

    # Tokenize the text
    logging.info(f"Tokenizing text: '{text}'")
    token_ids = tokenizer.encode(text)

    logging.info(f"Generated {len(token_ids)} tokens")
    logging.debug(f"Token IDs: {token_ids}")

    # Write to binary file
    # Format: uint32 for count, followed by uint32 for each token ID
    with open(output_file, 'wb') as f:
        # Write number of tokens
        f.write(struct.pack('<I', len(token_ids)))  # little-endian uint32

        # Write each token ID
        for token_id in token_ids:
            f.write(struct.pack('<I', token_id))  # little-endian uint32

    logging.info(f"Token IDs saved to {output_file}")

def main():
    if len(sys.argv) != 3:
        logging.error("Usage: python tokenize_text.py <text> <output_file>")
        logging.error("Example: python tokenize_text.py \"Hello world!\" tokens.bin")
        sys.exit(1)

    text = sys.argv[1]
    output_file = sys.argv[2]

    tokenize_and_save(text, output_file)

if __name__ == "__main__":
    main()