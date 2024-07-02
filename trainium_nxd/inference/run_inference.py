# Script for inference testing for DeepSeekCoder models (or other LLama-like LLMs)
# using AWS Trainium chips.

# No optimum.neuron is required, only transformers and neuronx_distributed


import transformers.modeling_utils as modeling_utils
from transformers_neuronx import LlamaForSampling
from transformers import AutoTokenizer
import torch_xla.core.xla_model as xm
import torch
import os
import time
from functools import reduce

CP_MODEL_DIR = "/home/ubuntu/dev/tp_dsk_ndx_pretrain/inference/inf_dsk_model"
CP_MODEL_PATH = "/home/ubuntu/dev/tp_dsk_ndx_pretrain/inference/inf_dsk_model/checkpoint.pt"
FULL_MODEL_PATH = "/home/ubuntu/dev/tp_dsk_ndx_pretrain/get_pretrained_model/dsk-1.3b-base-pretrained/step_2000/model"
# FULL_MODEL_PATH = "/home/ubuntu/dev/tp_dsk_ndx_pretrain/get_pretrained_model/dsk-1.3b-base-pretrained/step_2000"


# For PT autocast.
torch.cuda.is_bf16_supported = lambda: True

# Workaround for NaNs seen with transformers version >= 4.21.0
# https://github.com/aws-neuron/aws-neuron-sdk/issues/593


if os.environ.get("XLA_USE_BF16") or os.environ.get("XLA_DOWNCAST_BF16"):
    modeling_utils.get_parameter_dtype = lambda x: torch.bfloat16


def get_dtype(model) -> str:
    """
    Reference: https://pytorch.org/xla/release/1.12/index.html#xla-tensors-and-bfloat16
    """
    if "XLA_USE_BF16" in os.environ:
        return "torch.bfloat16"
    if "XLA_DOWNCAST_BF16" in os.environ:
        if "torch.float" in str(model.dtype):
            return "torch.bfloat16"
        if "torch.double" in str(model.dtype):
            return "torch.float32"
    return str(model.dtype)


def inference_test():

    is_root = xm.is_master_ordinal(local=False)

    # DSK-base 33b
    # MODEL_TO_USE, TOKENIZER_TO_USE, TP_DEGREE = \
    #     'deepseek-ai/deepseek-coder-33b-base',\
    #     'deepseek-ai/deepseek-coder-33b-base',\
    #     4

    # DSK-base 1.3b PRETRAINED
    MODEL_TO_USE, TOKENIZER_TO_USE, TP_DEGREE = \
        '/home/ubuntu/dev/tp_dsk_ndx_pretrain/inference/sft_dsk13',\
        'deepseek-ai/deepseek-coder-1.3b-base',\
        8

    CONTEXT_LEN = 100
    TO_GENERATE = 500

    N_POS = CONTEXT_LEN + TO_GENERATE

    if is_root:
        print("*"*10, " Loading from pretrained")
    neuron_model = LlamaForSampling.from_pretrained(
        MODEL_TO_USE,
        n_positions=N_POS,
        batch_size=1,
        tp_degree=TP_DEGREE,
        amp='f16'
    )
    if is_root:
        print("*"*10, " Loading to neuron")
    neuron_model.to_neuron()

    if is_root:
        print("*"*10, " Getting tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_TO_USE)

    if is_root:
        print("--------MODEL CONFIG----------")
        print(neuron_model.config)
        print("---------------------------------")

    def inference(model: LlamaForSampling, question):
        # input_ids = tokenizer.encode(question, return_tensors='pt')
        input_ids = question

        with torch.inference_mode():
            start = time.time()
            generated_samples = model.sample(
                input_ids, sequence_length=N_POS, top_k=50
            )
            elapsed = time.time() - start
            print(f"{type(generated_samples)=}{generated_samples.shape=}")
            response = "".join([tokenizer.decode(seq, skip_special_tokens=True) for seq in generated_samples])
            input_length = len(input_ids[0])
            output_length = len(generated_samples[0])
            no_generated_tokens = output_length - input_length
            tokens_per_second = no_generated_tokens / elapsed
            xm.master_print(
                f'generated sequences (len={output_length}) in {elapsed} seconds ({tokens_per_second}tokens/s)', flush=True)
        return {
            "output": response,
            "no_total_tokens": output_length,
            "no_input_tokens": input_length,
            "no_generated_tokens": no_generated_tokens,
            "time": elapsed,
            "tok_per_sec": tokens_per_second
        }

    def generate_test_data(tokenizer, start_token=1000, end_token=5000, context_length=100, no_samples=10):
        return [torch.randint(start_token, end_token, (1, context_length)) for _ in range(no_samples)]


# manually provided list of questions
#     MY_QUESTIONS = [
#         "// Add a function to add two integers and return the output in C++",
#         """/*****************************************************************************
# *
# *  COPYRIGHT (C) by Ericsson AB
# *
# *  The copyright to the computer program(s) herein is the property
# *  of Ericsson AB.
# *
# *  The program(s) may be used and/or copied only with the written
# *  permission from Ericsson AB or in accordance with the terms
# *  and conditions stipulated in the agreement/contract under which
# *  the program(s) have been supplied.
# *
# *
#         """,
#         "import socket\n\ndef ping_exponential_backoff(host: str):"
#     ]

    MY_QUESTIONS = generate_test_data(
        tokenizer=tokenizer,
        start_token=1000,
        end_token=3000,
        context_length=CONTEXT_LEN,
        no_samples=10
    )
    xm.master_print("Question loop begins:", flush=True)

    results = []

    for idx, q in enumerate(MY_QUESTIONS):
        xm.master_print(
            "Current question ({}): {}".format(idx, q),
            flush=True,
        )
        output = inference(neuron_model, q)
        xm.master_print(f"ANSWER: {output.pop('output')}")
        results.append(output)
    mean_tok_s = reduce(lambda x, y: x + y, [d['tok_per_sec'] for d in results])/len(results)
    xm.master_print(f"FINISHED. Mean tok/s: {mean_tok_s}. Results: {results}")


def main():
    torch.set_default_tensor_type("torch.FloatTensor")
    inference_test()


if __name__ == "__main__":
    main()
