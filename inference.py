#!/usr/bin/env python3
"""
Process tokens with OLMo-2-0425-1B model using transformers library.
Reads binary token files in the same format as the C++ code.
"""

import sys
import struct
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def read_tokens(file_path):
    """Read uint32 tokens from binary file."""
    with open(file_path, 'rb') as f:
        data = f.read()

    # Each token is 4 bytes (uint32)
    num_tokens = len(data) // 4
    tokens = []
    for i in range(num_tokens):
        token = struct.unpack('<I', data[i*4:(i+1)*4])[0]
        tokens.append(token)

    return tokens


def main():
    if len(sys.argv) < 2:
        print("Usage: inference.py <token_file1> [token_file2 ...]", file=sys.stderr)
        sys.exit(1)

    file_paths = sys.argv[1:]

    # Load model and tokenizer
    model_name = "allenai/OLMo-2-0425-1B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float32
    )
    model.eval()

    # Read all token files
    all_tokens = []
    for file_path in file_paths:
        tokens = read_tokens(file_path)
        all_tokens.append(tokens)

    # Find max sequence length and build batch with padding
    max_seq_len = max(len(tokens) for tokens in all_tokens)
    pad_token_id = tokenizer.pad_token_id
    batch = torch.full((len(all_tokens), max_seq_len), pad_token_id, dtype=torch.long)

    for i, tokens in enumerate(all_tokens):
        batch[i, :len(tokens)] = torch.tensor(tokens, dtype=torch.long)

    # Forward pass
    with torch.no_grad():
        outputs = model(batch)
        logits = outputs.logits

    # Print logits
    print(logits.shape)
    print(logits)

if __name__ == "__main__":
    main()
