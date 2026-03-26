#!/usr/bin/env python3

import argparse
from transformers import AutoModelForCausalLM
import os
import numpy as np
import logging


def main():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(description="Convert a Hugging Face model checkpoint.")
    parser.add_argument("model_name", help='Hugging Face model id, e.g. "allenai/OLMo-2-0425-1B"')
    parser.add_argument("output_dir", help="Output directory for converted artifacts")
    args = parser.parse_args()

    model_name = args.model_name
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    olmo = AutoModelForCausalLM.from_pretrained(model_name)
    # tokenizer = AutoTokenizer.from_pretrained(model_name)

    for name, parameter in olmo.named_parameters():
        # Create a safe filename from parameter name
        safe_name = name.replace('/', '_')
        filepath = os.path.join(output_dir, f"{safe_name}.npy")

        # Convert to float32 numpy array and save in .npy format
        tensor_data = parameter.detach().cpu().float().numpy()
        np.save(filepath, tensor_data)

        logger.info(f"{name}\t{tuple(parameter.shape)}\t{filepath}")


if __name__ == "__main__":
    main()