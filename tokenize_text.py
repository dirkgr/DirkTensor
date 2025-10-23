#!/usr/bin/env python3
"""
Tokenize text using the dolma2 tokenizer and write token IDs as binary integers.
"""

import sys
import struct
import logging
import argparse
from transformers import AutoTokenizer

# Configure logging to stderr (bash-friendly)
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s',
    stream=sys.stderr
)

def tokenize_and_save(text: str, output_file: str = None):
    """Tokenize text and save token IDs as binary integers."""

    # Load the dolma2 tokenizer
    logging.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("allenai/dolma2-tokenizer")

    # Tokenize the text
    logging.info(f"Tokenizing text: '{text}'")
    token_ids = tokenizer.encode(text)

    logging.info(f"Generated {len(token_ids)} tokens")
    logging.debug(f"Token IDs: {token_ids}")

    # Determine output destination
    if output_file is None:
        # Write to stdout
        output_stream = sys.stdout.buffer
        logging.info("Writing token IDs to stdout")
    else:
        # Write to file
        output_stream = open(output_file, 'wb')
        logging.info(f"Writing token IDs to {output_file}")

    try:
        # Write each token ID as uint32 (little-endian)
        for token_id in token_ids:
            output_stream.write(struct.pack('<I', token_id))

        output_stream.flush()
    finally:
        # Close file if we opened one (don't close stdout)
        if output_file is not None:
            output_stream.close()

def main():
    parser = argparse.ArgumentParser(
        description='Tokenize text using the dolma2 tokenizer and write token IDs as binary integers.',
        epilog='Examples:\n'
               '  %(prog)s "Hello world!" tokens.bin  # Write to file\n'
               '  %(prog)s "Hello world!"             # Write to stdout',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('text',
                        help='Text to tokenize')
    parser.add_argument('output_file',
                        nargs='?',
                        default=None,
                        help='Output file for token IDs (default: write to stdout)')

    args = parser.parse_args()

    tokenize_and_save(args.text, args.output_file)

if __name__ == "__main__":
    main()