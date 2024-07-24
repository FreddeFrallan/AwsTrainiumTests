# Note: transformers==4.36.1 works here for me, it might be because for pretrainining I was using 4.31.0.
# You want 4.36.1 for this script, as well as for the inference script.

import argparse
from transformers import AutoModelForCausalLM, AutoConfig
import torch

def main(args):
    print(f"Loading model configfrom HF: {args.model_path}", flush=True)
    cfg = AutoConfig.from_pretrained(args.model_path)

    print(f"Creating model from config: {cfg}", flush=True)
    model = AutoModelForCausalLM.from_config(cfg)

    print(f"Loading checkpoint from {args.checkpoint_location}", flush=True)
    state_dict = torch.load(args.checkpoint_location)

    print(f"Updating model with weights from the checkpoint", flush=True)
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

    if missing_keys:
        print(f"ERROR: missing keys in the state dict: {missing_keys}")

    if unexpected_keys:
        print(f"WARNING: unexpected keys in the state dict: {unexpected_keys}")

    print(f"Saving the model to {args.output_dir}", flush=True)
    model.save_pretrained(args.output_dir)

    print("DONE!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process model, checkpoint, and output directory paths.")

    # Add the arguments
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model (HF))')
    parser.add_argument('--checkpoint_location', type=str, required=True, help='Path to the checkpoint file (.pt)')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save the output')
    
    args = parser.parse_args()
    main(args)