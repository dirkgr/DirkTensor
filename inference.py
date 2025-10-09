#!/usr/bin/env python3
"""
Process tokens with OLMo-2-0425-1B model using transformers library.
Reads binary token files in the same format as the C++ code.
"""

import sys
import struct
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def read_tokens(file_path=None):
    """Read uint32 tokens from binary file or stdin."""
    if file_path:
        with open(file_path, 'rb') as f:
            data = f.read()
    else:
        data = sys.stdin.buffer.read()

    # Each token is 4 bytes (uint32)
    num_tokens = len(data) // 4
    tokens = []
    for i in range(num_tokens):
        token = struct.unpack('<I', data[i*4:(i+1)*4])[0]
        tokens.append(token)

    return tokens


def main():
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        file_path = None

    # Read tokens
    tokens = read_tokens(file_path)
    print(f"Read {len(tokens)} tokens")

    # Load model and tokenizer
    model_name = "allenai/OLMo-2-0425-1B"
    print(f"Loading model {model_name}...")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float32
    )
    model.eval()

    # Process each token
    with torch.no_grad():
        for i, token in enumerate(tokens):
            print(f"token {token}")

            # Create input tensor with all tokens up to and including current one
            input_ids = torch.tensor([tokens[:i+1]], dtype=torch.long)

            # Get model predictions
            outputs = model(input_ids)
            logits = outputs.logits[0, -1, :]  # Get logits for the last position

            # Get top 5 predictions
            top5_logits, top5_indices = torch.topk(logits, 5)
            top5_tokens = top5_indices.cpu().numpy()

            print(f"Top 5 next tokens: {' '.join(map(str, top5_tokens))}")


if __name__ == "__main__":
    main()
