#!/usr/bin/env python3
"""
Benchmark forward + backward pass with OLMo-2-0425-1B model.
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
    num_tokens = len(data) // 4
    tokens = [struct.unpack('<I', data[i*4:(i+1)*4])[0] for i in range(num_tokens)]
    return tokens


def main():
    if len(sys.argv) < 2:
        print("Usage: benchmark.py <token_file>", file=sys.stderr)
        sys.exit(1)

    file_path = sys.argv[1]

    # Load model
    print("Loading model...", file=sys.stderr)
    model_name = "allenai/OLMo-2-0425-1B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)

    # Enable gradients
    for param in model.parameters():
        param.requires_grad_(True)

    # Read tokens
    tokens = read_tokens(file_path)
    batch = torch.tensor([tokens], dtype=torch.long)

    print(f"Tokens: {len(tokens)}", file=sys.stderr)

    # Warm-up run (optional, but helps with JIT compilation)
    # Skip warmup to compare apples to apples with C++

    # Timed run
    torch.cuda.synchronize() if torch.cuda.is_available() else None

    start = time.perf_counter()

    # Forward pass
    outputs = model(batch)
    logits = outputs.logits

    forward_done = time.perf_counter()

    # Compute loss
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = batch[:, 1:].contiguous()
    loss = torch.nn.functional.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=tokenizer.pad_token_id,
    )

    print(f"Loss: {loss.item()}")

    # Backward pass
    loss.backward()

    end = time.perf_counter()

    torch.cuda.synchronize() if torch.cuda.is_available() else None

    forward_ms = (forward_done - start) * 1000
    total_ms = (end - start) * 1000
    backward_ms = total_ms - forward_ms

    print(f"Tokens: {len(tokens)}")
    print(f"Forward:  {forward_ms:.1f} ms")
    print(f"Backward: {backward_ms:.1f} ms")
    print(f"Total:    {total_ms:.1f} ms")


if __name__ == "__main__":
    main()
