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

def tokenize_and_save(text: str, output_file: str = None):
    """Tokenize text and save token IDs to binary file or stdout."""

    # Load the dolma2 tokenizer
    logging.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("allenai/dolma2-tokenizer")

    # Tokenize the text
    logging.info(f"Tokenizing text: '{text}'")
    token_ids = tokenizer.encode(text)

    logging.info(f"Generated {len(token_ids)} tokens")
    logging.debug(f"Token IDs: {token_ids}")

    # Determine output destination
    if output_file is None or output_file == '-':
        # Write to stdout
        output_stream = sys.stdout.buffer
        logging.info("Writing token IDs to stdout")
    else:
        # Write to file
        output_stream = open(output_file, 'wb')
        logging.info(f"Writing token IDs to {output_file}")

    try:
        # Write binary data
        # Format: uint32 for count, followed by uint32 for each token ID
        output_stream.write(struct.pack('<I', len(token_ids)))  # little-endian uint32

        # Write each token ID
        for token_id in token_ids:
            output_stream.write(struct.pack('<I', token_id))  # little-endian uint32

        output_stream.flush()
    finally:
        # Close file if we opened one (don't close stdout)
        if output_file is not None and output_file != '-':
            output_stream.close()

def main():
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        logging.error("Usage: python tokenize_text.py <text> [output_file]")
        logging.error("Examples:")
        logging.error("  python tokenize_text.py \"Hello world!\" tokens.bin  # Write to file")
        logging.error("  python tokenize_text.py \"Hello world!\" -           # Write to stdout")
        logging.error("  python tokenize_text.py \"Hello world!\"             # Write to stdout")
        sys.exit(1)

    text = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) == 3 else None

    tokenize_and_save(text, output_file)

if __name__ == "__main__":
    main()