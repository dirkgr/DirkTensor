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


def backward_hook(module, grad_input, grad_output):
    """Backward hook - no-op for debugger attachment."""
    pass  # Set breakpoint here to inspect grad_output


def register_backward_hooks(model):
    """Register pre-backward hooks on all modules for debugging."""
    for name, module in model.named_modules():
        module.register_full_backward_hook(backward_hook)


def print_gradient_slice(name, grad):
    """Print first 5 values along each dimension of a gradient tensor."""
    if grad is None:
        print(f"{name}: No gradient")
        return

    print(f"{name}: shape={list(grad.shape)}")

    # Slice to first 5 along each dimension
    slices = tuple(slice(0, min(5, s)) for s in grad.shape)
    sliced = grad[slices]
    print(sliced)
    print()


def main():
    if len(sys.argv) < 2:
        print("Usage: backward.py <token_file1> [token_file2 ...]", file=sys.stderr)
        sys.exit(1)

    file_paths = sys.argv[1:]

    # Load model and tokenizer
    model_name = "allenai/OLMo-2-0425-1B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float32
    )

    # Enable gradients for all parameters
    for param in model.parameters():
        param.requires_grad_(True)

    # Register pre-backward hooks for debugging
    register_backward_hooks(model)

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

    # Forward pass (no torch.no_grad() so we can compute gradients)
    outputs = model(batch)
    logits = outputs.logits

    # Print logits shape
    print("Logits shape:", logits.shape)

    # Compute loss for backward pass (cross-entropy with shifted labels)
    # Shift logits and labels for next-token prediction
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = batch[:, 1:].contiguous()

    # Flatten for cross-entropy
    loss = torch.nn.functional.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=pad_token_id,
        )

    print(f"Loss: {loss.item()}")

    # Backward pass
    loss.backward()

    # Print gradients for all model parameters
    print("\n" + "="*80)
    print("GRADIENTS")
    print("="*80 + "\n")

    for name, param in model.named_parameters():
        print_gradient_slice(name, param.grad)

if __name__ == "__main__":
    main()
