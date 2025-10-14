#!/usr/bin/env python3
"""Export tokenizer vocabulary to a simple text format for C++ to read."""

import argparse
import logging
from transformers import AutoTokenizer


def main():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(
        description="Export tokenizer vocabulary to a simple text format for C++ to read."
    )
    parser.add_argument(
        "model_name",
        help='Hugging Face model id, e.g. "allenai/OLMo-2-0425-1B"'
    )
    parser.add_argument(
        "output_file",
        help="Output vocabulary file path"
    )
    args = parser.parse_args()

    # Load the tokenizer
    logger.info(f"Loading tokenizer from {args.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # Get the vocabulary
    vocab = tokenizer.get_vocab()

    # Sort by token ID
    sorted_vocab = sorted(vocab.items(), key=lambda x: x[1])

    # Write to file
    logger.info(f"Writing {len(sorted_vocab)} tokens to {args.output_file}...")
    with open(args.output_file, "w", encoding="utf-8") as f:
        for token, idx in sorted_vocab:
            # Escape newlines and backslashes for simple parsing
            token = token.replace("\\", "\\\\").replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")
            f.write(f"{idx}\t{token}\n")

    logger.info(f"Successfully exported {len(sorted_vocab)} tokens to {args.output_file}")


if __name__ == "__main__":
    main()
