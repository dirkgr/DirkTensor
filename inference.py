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

    # Load model and tokenizer
    model_name = "allenai/OLMo-2-0425-1B"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float32
    )
    model.eval()

    tokens_left = 20
    next_token_id = 0
    past_key_values = None

    # Process input tokens
    with torch.no_grad():
        for token_id in tokens:
            decoded = tokenizer.decode([token_id])
            print(f'token {token_id} ("{decoded}") ', end='')

            # Forward pass with single token
            input_ids = torch.tensor([[token_id]], dtype=torch.long)
            outputs = model(input_ids, past_key_values=past_key_values, use_cache=True)
            past_key_values = outputs.past_key_values
            logits = outputs.logits[0, -1, :]

            # Get top 5 predictions
            top5_logits, top5_indices = torch.topk(logits, 5)
            top5_tokens = top5_indices.cpu().numpy()

            print("Top 5 next tokens: ", end='')
            for top_token in top5_tokens:
                decoded_top = tokenizer.decode([top_token])
                print(f'{top_token} ("{decoded_top}") ', end='')
            print()

            next_token_id = top5_tokens[0]
            tokens_left -= 1

        # Continue generation
        print(tokenizer.decode([next_token_id]), end='', flush=True)

        while tokens_left > 0:
            input_ids = torch.tensor([[next_token_id]], dtype=torch.long)
            outputs = model(input_ids, past_key_values=past_key_values, use_cache=True)
            past_key_values = outputs.past_key_values
            logits = outputs.logits[0, -1, :]

            top5_logits, top5_indices = torch.topk(logits, 5)
            next_token_id = top5_indices[0].item()

            print(tokenizer.decode([next_token_id]), end='', flush=True)
            tokens_left -= 1

        print()


if __name__ == "__main__":
    main()
