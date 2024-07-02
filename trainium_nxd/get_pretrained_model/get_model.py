import torch
# from transformers.models.llama.modeling_llama import LlamaForCausalLM
# model = LlamaForCausalLM.from_pretrained("NousResearch/Llama-2-7b-hf")
# torch.save(model.state_dict(), "llama-7b-hf-pretrained.pt")

# WHICH MODEL TO GET AND WHERE TO SAVE IT
MODEL_BASE = "deepseek-ai/deepseek-coder-6.7b-base"
MODEL_SAVE_PATH = "deepseek-coder-6.7b-base-pretrained.pt"
CONFIG_SAVE_PATH = "config-dsk-6.7b.json"

from transformers import AutoModelForCausalLM, AutoConfig

print(f"Loading model form HF: {MODEL_BASE}")
model = AutoModelForCausalLM.from_pretrained(MODEL_BASE)
print(f"Saving model to {MODEL_SAVE_PATH}")
torch.save(model.state_dict(), MODEL_SAVE_PATH)

print(f"Loading config from HF: {MODEL_BASE}")
cfg = AutoConfig.from_pretrained(MODEL_BASE)
print(f"Saving model config to {CONFIG_SAVE_PATH}")
cfg.to_json_file(CONFIG_SAVE_PATH)

# later, convert checkpoitns with:
# python3 convert_checkpoints.py --tp_size 2 --convert_from_full_model --coalesce_qkv true --config config-dsk-1.3b-base.json --input_dir deepseek-coder-1.3b-base-pretrained.pt --output_dir dsk-1.3b-base-pretrained/pretrained_weight
# note, that coalesce_qkv is a bool, so if you don't want it, remove it completly instead of giving coalesce_qkv false
