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
    """Backward hook - prints gradient info for specific modules."""
    pass  # Set breakpoint here to inspect grad_output


def mlp_backward_hook(module, grad_input, grad_output):
    """Print gradient entering MLP backward."""
    if grad_output[0] is not None:
        print(f"MLP backward d_output first 5 values: {grad_output[0][0, 0, :5].tolist()}")


def lm_head_backward_hook(module, grad_input, grad_output):
    """Print gradient entering and leaving LM head backward."""
    if grad_output[0] is not None:
        print(f"d_output entering backward first 5: {grad_output[0][0, 0, :5].tolist()}")
    if grad_input[0] is not None:
        print(f"grad after lm_head first 5: {grad_input[0][0, 0, :5].tolist()}")


def norm_backward_hook(module, grad_input, grad_output):
    """Print gradient leaving norm backward."""
    if grad_input[0] is not None:
        print(f"grad after norm first 5: {grad_input[0][0, 0, :5].tolist()}")


def make_block_backward_hook(layer_idx):
    """Create a backward hook for a specific block."""
    def hook(module, grad_input, grad_output):
        if grad_output[0] is not None:
            print(f"Block {layer_idx} backward input first 5: {grad_output[0][0, 0, :5].tolist()}")
        if grad_input[0] is not None:
            print(f"Block {layer_idx} backward output first 5: {grad_input[0][0, 0, :5].tolist()}")
    return hook


def make_layer15_component_hook(name):
    """Create hooks for layer 15 components."""
    def hook(module, grad_input, grad_output):
        if grad_output and len(grad_output) > 0 and grad_output[0] is not None:
            print(f"  {name} grad_output first 5: {grad_output[0][0, 0, :5].tolist()}")
        if grad_input and len(grad_input) > 0 and grad_input[0] is not None:
            print(f"  {name} grad_input first 5: {grad_input[0][0, 0, :5].tolist()}")
    return hook


def register_backward_hooks(model):
    """Register pre-backward hooks on all modules for debugging."""
    for name, module in model.named_modules():
        module.register_full_backward_hook(backward_hook)
        if name == "model.layers.0.mlp":
            module.register_full_backward_hook(mlp_backward_hook)
        if name == "lm_head":
            module.register_full_backward_hook(lm_head_backward_hook)
        if name == "model.norm":
            module.register_full_backward_hook(norm_backward_hook)
        # Register block hooks for layers 15, 14, 13
        for i in [15, 14, 13, 1, 0]:
            if name == f"model.layers.{i}":
                module.register_full_backward_hook(make_block_backward_hook(i))
        # Register hooks for layer 15 components
        if name == "model.layers.15.post_feedforward_layernorm":
            module.register_full_backward_hook(make_layer15_component_hook("postMlpNorm"))
        if name == "model.layers.15.mlp":
            module.register_full_backward_hook(make_layer15_component_hook("mlp"))
        if name == "model.layers.15.post_attention_layernorm":
            module.register_full_backward_hook(make_layer15_component_hook("postAttentionNorm"))
        if name == "model.layers.15.self_attn":
            module.register_full_backward_hook(make_layer15_component_hook("attention"))


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


def compute_loss(model, batch, pad_token_id):
    """Forward pass and compute cross-entropy loss."""
    outputs = model(batch)
    logits = outputs.logits

    # Shift logits and labels for next-token prediction
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = batch[:, 1:].contiguous()

    # Flatten for cross-entropy
    loss = torch.nn.functional.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=pad_token_id,
    )
    return loss


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

    # First forward pass
    loss = compute_loss(model, batch, pad_token_id)
    print(loss.item())

    # Backward pass
    loss.backward()

    # Optimizer step (SGD with lr=1e-4)
    with torch.no_grad():
        for param in model.parameters():
            if param.grad is not None:
                param -= 1e-4 * param.grad

    # Zero gradients
    model.zero_grad()

    # Second forward pass
    loss = compute_loss(model, batch, pad_token_id)
    print(loss.item())

if __name__ == "__main__":
    main()
