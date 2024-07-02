# Training LLMs on the Trainium instances

A guide on how to use [Trainium](https://aws.amazon.com/machine-learning/trainium/) instances for LLM training.

There are two instance types which you can choose:
- trn1.2xlarge: 8vCPUs, 1 Neuron Device (2 cores, 16 GB per core - 32GB total)
- trn1.32xlarge: 128vCPUs, 16 Neuron Device (32 cores, 512 GB per core - 32GB total)

The smaller one is supposed to be better for testing your setup as it's much cheaper (1.33$ vs 21.5$ when writing this readme), and the larger one for high-scale training. However, keep in mind that the larger models may not fit the 32GB NeuronDevice memory.

**NOTE**:
For this guide, only trn1.32xlarge was used as it was free for the 6GTB02 account anyway, but there should be no difference in the general setup with the smaller trn instance.

# How to set up a trainium instance in AWS


## Setting up the instance using pure Ubuntu 22.04 image (unstable)

An official guide on how to setup torch-neuronx for training on the Trainium instance can be found [here](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/setup/torch-neuronx.html#setup-torch-neuronx).


# Run precompilation & training

There are currently two frameworks we tested with trainium:

## optimum-neuron

An official HuggingFace portal for [optiumum-neuron](https://huggingface.co/docs/optimum-neuron/index).

This framework implements all the low-level operations under the hood, and gives the user the NeuronTrainer class which has a very similar (almost the same?) API as the standard Trainer class from transformers library.

It is currently under heavy development, and we were not able to fully use it for our use case:
Train the deepseekcoder models (1.3b, 6.7b, 33b) with tensor parallelism, using custom loss function and evaluation function. 

If you want to take a look at the code we tested, you can find it in __optimum_neuron_hf__ directory. Feel free to take the scripts and modify according to your needs!

### Example

In order to start training, the models files need to be precompiled. This is a long process (~10-40min depending on model size). A caching mechanism is available but currently we only use local cache. In order to set it up (and other helpful flags), export the following in the shell before running precompilation&training:

`export MALLOC_ARENA_MAX=64`
`export NEURON_CC_FLAGS="--model-type=transformer --distribution-strategy=llm-training --cache_dir=/home/ubuntu/neuron_compile_cache/ -O1"`

Precompilation is needed for each setup: (model, batch, tensor_parralel, some other scripts args). Meaning that it needs to run each time it doesn't find precompiled model with given set of parameters. It will run automatically when you call the training script, but you can run precompilation-only using the following command:

`neuron_parallel_compile torchrun --nproc_per_node=32 run_train_and_eval.py <precompilation_args_path> 2>&1 | tee log_precompilation.txt`

* (https://pytorch.org/docs/stable/elastic/run.html)[trochrun] is a convinient way of running torch.distributed.launch
* __run_train_and_eval.py__ is the trainig script.
* --nproc_per_node=32 - defines how many Cores to use
* __precompilation_args_path__ points to the script args for precompilation. See __deepseekcoder_*__ for examples. The YAML files are annotated with my comments.
* the remainig part tees the output to the logfile and can be skipped

Once the precompilation is done, the results (NEFF files) should be saved in the --cache_dir (/home/ubuntu.neuron_compile_cache). To start training, run:

`torchrun --nproc_per_node=32 run_train_and_eval.py <training_args_path> 2>&1 | tee log_training.txt`

When the training is running, you can use `neuron-top` to see if the NeuronCores are utilized properly.

## neuronx-distributed

An official AWS Neuron Documentation for [neuronx-distributed](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/libraries/neuronx-distributed/index.html)

This is a lower level library which requires replacing orginal (hf transformers) layers of the model with proper parallel implementations. For some models (i.e., llama based) this has already beed done by AWS and can be reused.

It supports tensor parallelism and pipeline parallelism (PP not verified yet), as well as ZeRO_1 and some other features that improve training speed or reduce memory usage (check the docs).

### Training

#### Modelling LLM

In order to train the model from scratch, you just need a config file:

```python
from transformers import AutoConfig
cfg = AutoConfig.from_pretrained("path_to_model")
```

That config can be used to initialize empty model implemented using neuronx-distributed (NxD) parallel layers. Example can be found in __trainium_nxd/modelling_llama_nxd.py__ which is used in __trainium_nxd/tp_zero1_dsk_run_eval.py__ in `get_model()` function. For other types of models, check AWS documentation, if they are available they can be reused, otherwise you need to implement it yourself.

#### Continued pretraining

For continued pretraining, you need to load a model with pretrained weights (you still need the modelling part described above).

You can use scripts from __trainium_nxd/get_pretrained_model/convert_checkpoints.py__ to fetch the model and convert it to trainium consumable format:

1. Use __get_model.py__ to fetch desired model and store it locally along with the confg.json file
2. Use __convert_checkpoints.py__ to convert the model to desired format:

Example:

```bash 
python3 convert_checkpoints.py 
    --tp_size 2 
    --convert_from_full_model 
    --coalesce_qkv true 
    --config config-dsk-1.3b-base.json 
    --input_dir deepseek-coder-1.3b-base-pretrained.pt 
    --output_dir dsk-1.3b-base-pretrained/pretrained_weight
```

where **tp_size** is the tensor_parallel degree you want to use for training. Note, that if you want to use different tp_size, you need to use convert_checkpoints once again, to shard the model in a proper way.

**coalesce_qkv** is a boolean flag, if you don't need it - remove it completly. Especially for smaller models, the Q, K, and V attention matrices are often concatenated into a qkv matrix, and if this is the case for your model, you need to do use this flag. You can check it by loading a model state dict (.pt file created by __get_model.py__) and listing the keys. If they contain **qkv_...** in the layers, that means in the orginal model matrices are coalesced and you need to do it as well. If you see separate entries for q, k and v - you don't need this flag.


The main training script is __traininium_nxd/tp_zero1_dsk_run_eval.py__ and the bash script that handles all the flags is __trainium_nxd/tp_zero1_dsk_hf_pretrain.sh__. There are comments in the bash script, so go throuhg it and see what you need to change. For example to use the pretrained model that was created in the previous step, find **PRETRAINED_WEIGHT** in the script and set proper value.

#### Precompilation

The full training consists of two parts: precompilation and training. Precompilation is done once per model with a specific set of parameters (tp, pp, etc.). Precompiled models are stored in **cache_dir** pointed in the bash script in **NEURON_CC_FLAGS** (compiler flags). The precompilation process can last 10-40 min depending on the model size. In order to run precompilation, use:

```bash
neuron_parallel_compile ./tp_zero1_dsk_hf_pretrain.sh
```

It will automatically limit the number of steps, precompile the entire model and exit. This is not mandatory, but if you skip this part, the precompilation will run when you start the training. If the precompilation fails, its eigher something wrong with the script (you will see python errors), or it might be that the model can't fit the devices. In this case you will see Neuron SDK errors, and you need to increase TP or PP if possible.


#### Start training

In order to start the training, simply run

```bash
./tp_zero1_dsk_hf_pretrain.sh
```

Note, that all the parameters must be set in the script. The most important ones are:

* TP_DEGREE - tensor parallel degree - same value as you choose your model to be when sharding
* GBS - global batch size - this controls gradient accumulation. It also includes DP (below)
* NUM_NEURONCORES and NEURON_RT_NUM_CORES - how many Neuron Cores to use (trn1.32xlarge has 32)
* TOTAL_STEPS, EVAL_STEPS - step = weight activation which happens once every GBS iterations (including DP)

DP - Data Parallel size. You can find these two lines in the script:

```bash
DP=$(($NEURON_RT_NUM_CORES * $WORLD_SIZE / $TP_DEGREE))
ACC_STEPS=$(($GBS / $MBS / $DP))
# WORLD_SIZE and MBS are always 1 here
```

Meaning, that if you shard your model across TP_DEGREE neuron cores, and you use NEURON_RT_NUM_CORES cores in total,
the script will make DP copies of the model, feed it with different data and synchronize gradiends and weights.

ACC_STEPS are accumulation steps, so if you have DP copies of your model, each of them need to perform ACC_STEPS iterations to achieve GBS and trigger backpropagation.


# Inference

## Sharded model consolidation

If you trained the model on the trainium and it is saved as a checkpoint, you need to consolidate it first. You can use __trainium_nxd/inference/convert_checkpoints.py__ script. Example:

```bash
python3 convert_checkpoints.py 
    --tp_size 2 
    --convert_to_full_model 
    --config config-dsk-1.3b-base.json 
    --input_dir ../get_pretrained_model/dsk-1.3b-base-pretrained/step_2000/model 
    --output_dir inf_dsk_model 
    --coalesce_qkv true 
    --load_xser true
```

Note, that this time you need convert_to_full_model flag. Also, if your checkpoints were saved with xser (serialized pytorch which is true by default in the training script), you need to add load_xser flag.

When done, you will end up with a single checkpoint.pt file which is a state dict for your model. In order to save the model in a HF consumable format (i.e., safetensors), you need to do the following (python example):

```python
from transformers import AutoModelForCausalLM
import torch

model = AutoModelForCausalLM.from_pretrained("your_model")
state_dict = torch.load('checkpoint.pt')

model.load_state_dict(state_dict)

model.save_pretrained("your_directory")
```

## optimum-neuron

Something we did not try, follow official guide: https://huggingface.co/docs/optimum-neuron/guides/export_model


## transformers-neuronx

For inference, you can write a script based on __trainium_nxd/inference/run_inference.py__. It shows how to load the model, tokenizer and use specified TP and other parameters. Note, that for inference you don't need neuronx-distributed, this time you need transformers-neuronx which depends on a higher version of transformers library. You can use the preinstalled python venv for inference: 

```bash
source /opt/aws_neuronx_venv_transformers_neuronx/bin/activate
```

You also do not need to run it on several nodes (torchrun not needed!), so in order to start the script you simply run it with:

```bash
python3 run_inference.py
```

# Known issues

1. For optimum-neuron, custom evaluation and custom metrics function does not work - even when properly provided, the NeuronTrainer ignores it and uses the default methods
2. For NxD the MBS must be 1, otherwise the precompilation will fail
3. Some topologies (DP=8 for instance, or TP=16) are not supported for the NxD, read the error messages carefully
4. If the precompilation keeps failing, clear the cache_dir manually, and clear the locks:

```bash
neuron_parallel_compile --command clear-locks
neuron_parallel_compile --command clean
```

5. If the neuron devices are not responding, check if some process is using them:

```bash
lsmod |grep neuron
```

The last column in the output should be 0: 

neuron                299008  0

If its not, find and kill all the processes that might be using the neuron cores.

If this does not help, restart the neuron service and check again:

```bash
sudo rmmod neuron
sudo modprobe neuron
```

7. Usefull tips for troubleshooting: [link](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/neuron-runtime/nrt-troubleshoot.html#)

