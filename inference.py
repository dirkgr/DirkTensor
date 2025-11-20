#!/usr/bin/env python3
"""
Process tokens with OLMo-2-0425-1B model using transformers library.
Reads binary token files in the same format as the C++ code.
"""

import sys
import struct
import time
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

    # Track total time
    total_start = time.perf_counter()

    # Load model and tokenizer
    model_name = "allenai/OLMo-2-0425-1B"

    print(f"Loading model {model_name}...", file=sys.stderr)
    model_load_start = time.perf_counter()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float32
    )
    model.eval()

    model_load_time = time.perf_counter() - model_load_start
    print(f"Model loaded in {model_load_time:.3f} seconds", file=sys.stderr)

    # Read all token files
    io_start = time.perf_counter()
    all_tokens = []
    total_tokens = 0
    for file_path in file_paths:
        tokens = read_tokens(file_path)
        all_tokens.append(tokens)
        total_tokens += len(tokens)
        print(f"Read {len(tokens)} tokens from {file_path}", file=sys.stderr)

    io_time = time.perf_counter() - io_start

    # Find max sequence length
    max_seq_len = max(len(tokens) for tokens in all_tokens)
    print(f"Max sequence length: {max_seq_len}, Total tokens: {total_tokens}", file=sys.stderr)

    # Build batch with padding
    batch_prep_start = time.perf_counter()
    pad_token_id = tokenizer.pad_token_id
    batch = torch.full((len(all_tokens), max_seq_len), pad_token_id, dtype=torch.long)

    for i, tokens in enumerate(all_tokens):
        batch[i, :len(tokens)] = torch.tensor(tokens, dtype=torch.long)

    batch_prep_time = time.perf_counter() - batch_prep_start

    # Warm-up run (optional, helps with more stable timing)
    print(f"\nRunning warm-up pass...", file=sys.stderr)
    with torch.no_grad():
        _ = model(batch)

    # Timed forward passes
    num_runs = 5
    print(f"\nRunning {num_runs} timed forward passes...", file=sys.stderr)
    forward_times = []

    for run in range(num_runs):
        torch.cuda.synchronize() if torch.cuda.is_available() else None

        forward_start = time.perf_counter()
        with torch.no_grad():
            outputs = model(batch)
            logits = outputs.logits
        torch.cuda.synchronize() if torch.cuda.is_available() else None

        forward_time = time.perf_counter() - forward_start
        forward_times.append(forward_time)
        print(f"  Run {run+1}: {forward_time:.4f} seconds", file=sys.stderr)

    # Calculate statistics
    avg_forward_time = sum(forward_times) / len(forward_times)
    min_forward_time = min(forward_times)
    max_forward_time = max(forward_times)

    # Calculate throughput
    tokens_per_sec = total_tokens / avg_forward_time

    total_time = time.perf_counter() - total_start

    # Print results
    print("\n" + "="*60, file=sys.stderr)
    print("PYTHON PERFORMANCE SUMMARY", file=sys.stderr)
    print("="*60, file=sys.stderr)
    print(f"Model loading:     {model_load_time:.3f} seconds", file=sys.stderr)
    print(f"File I/O:          {io_time:.4f} seconds", file=sys.stderr)
    print(f"Batch preparation: {batch_prep_time:.4f} seconds", file=sys.stderr)
    print(f"\nForward pass times ({num_runs} runs):", file=sys.stderr)
    print(f"  Average: {avg_forward_time:.4f} seconds", file=sys.stderr)
    print(f"  Min:     {min_forward_time:.4f} seconds", file=sys.stderr)
    print(f"  Max:     {max_forward_time:.4f} seconds", file=sys.stderr)
    print(f"\nThroughput: {tokens_per_sec:.1f} tokens/second", file=sys.stderr)
    print(f"Total time: {total_time:.3f} seconds", file=sys.stderr)
    print("="*60 + "\n", file=sys.stderr)

    # Print logits (original output)
    print(logits.shape)
    print(logits)

if __name__ == "__main__":
    main()
