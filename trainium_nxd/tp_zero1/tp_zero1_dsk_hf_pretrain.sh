#!/bin/bash

#############################################
# User defined parameters and env vars

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

export NEURON_CC_FLAGS="--model-type transformer --distribution-strategy=llm-training --cache_dir=/home/ubuntu/neuron_compile_cache/"
export NEURON_FUSE_SOFTMAX=1

# Async Runtime
export NEURON_RT_ASYNC_EXEC_MAX_INFLIGHT_REQUESTS=3

# HOST OOM
export MALLOC_ARENA_MAX=64

# TP degree
TP_DEGREE=2
# 0: bf16; 1: mixed precision
USE_MIX_PRECISION=1
# 0: use pure DP; 1: use ZeRO-1
USE_ZERO_1=1
# global batch size
: ${GBS:=128}
# micro batch size - DO NOT CHANGE THIS ONE - IT IS A KNOWN ISSUE THAT TRAINIUM FAILS TO DO MBS>1
MBS=1
# number of steps to run
TOTAL_STEPS=2500
# warmup steps
WARMUP_STEPS=10
# learning rate
LR=5.0e-5
# model path
# MODEL_PATH=$SCRIPT_DIR
MODEL_PATH="deepseek-ai/deepseek-coder-1.3b-base"
IGNORE_INDEX=32018
# data path
# DATA_PATH="~/examples_datasets/wikicorpus_llama2_7B_tokenized_4k"
DATA_PATH="/home/ubuntu/dev/train_dsk/raw_data_tokenized/training_v1"

# evaluation

DO_EVAL=1
EVAL_DATA_PATH="/home/ubuntu/dev/train_dsk/raw_data_tokenized/validation_v1" # eval_data_dir
EVAL_STEPS=100 # global steps, not microsteps

# sequence length
SEQ_LEN=512


### Checkpointing

CHECKPOINT_DIR="/home/ubuntu/dev/tp_dsk_ndx_pretrain/checkpoints"
CHECKPOINT_FREQ=500

# to disable loading checkpoint
LOADING_STEP=-1 

# or to start from latest checkpoint:
# LOADING_STEP="latest_if_exists" # to use latest step

# To load pretrained model (previously prepared by get_model and convert_checkpoints)
# (Coment all lines if not needed) 
# ---->START
# LOADING_STEP=-1 
# PRETRAINED_WEIGHT="true"
# CHECKPOINT_DIR="/home/ubuntu/dev/tp_dsk_ndx_pretrain/get_pretrained_model/dsk-1.3b-base-pretrained"
# <---- END

#############################################

export NUM_NEURONCORES=8
NODE_ID=0
WORLD_SIZE=1
DISTRIBUTED_ARGS="--nproc_per_node $NUM_NEURONCORES"
if [ ! -z "$SLURM_NTASKS" ]; then
    WORLD_SIZE=$SLURM_NTASKS
    NODE_ID=$SLURM_NODEID
    MASTER_ADDRESS=(`scontrol show hostnames $SLURM_JOB_NODELIST`)
    DISTRIBUTED_ARGS="--nproc_per_node $NUM_NEURONCORES --nnodes $WORLD_SIZE --node_rank $NODE_ID --master_addr $MASTER_ADDRESS --master_port 44000"
    if [ $NODE_ID -eq 0 ]; then
        echo "WORLD_SIZE=$WORLD_SIZE"
        echo "NODE_ID=$NODE_ID"
        echo "MASTER_ADDRESS=$MASTER_ADDRESS"
        echo "DISTRIBUTED_ARGS=$DISTRIBUTED_ARGS"
    fi
    export FI_EFA_USE_DEVICE_RDMA=1
    export FI_PROVIDER=efa
fi

echo "WORLD_SIZE=$WORLD_SIZE"
echo "NODE_ID=$NODE_ID"
echo "MASTER_ADDRESS=$MASTER_ADDRESS"

sudo sysctl -w net.ipv4.ip_local_reserved_ports=44000,48620

export NEURON_RT_NUM_CORES=8
export NUM_NEURONCORES=$NEURON_RT_NUM_CORES
export TPU_NUM_DEVICES=$NEURON_RT_NUM_CORES
export TPU_CHIPS_PER_HOST_BOUNDS=$NEURON_RT_NUM_CORES
export NEURON_RT_ROOT_COMM_ID=localhost:48620

#############################################

EXTRA_ARGS=" "
if [ $USE_MIX_PRECISION -gt 0 ]; then
    EXTRA_ARGS+=" --use_mix_precision"
fi
if [ $USE_ZERO_1 -gt 0 ]; then
    EXTRA_ARGS+=" --use_zero_1"
fi
if [ -n "${PRETRAINED_WEIGHT}" ]; then  
    echo "PRETRAINED_WEIGHT=$PRETRAINED_WEIGHT, adding to extra args"
    EXTRA_ARGS+=" --pretrained_weight"
fi

DP=$(($NEURON_RT_NUM_CORES * $WORLD_SIZE / $TP_DEGREE))
ACC_STEPS=$(($GBS / $MBS / $DP))


if [ -n $NEURON_EXTRACT_GRAPHS_ONLY ] && [ $NEURON_EXTRACT_GRAPHS_ONLY -gt 0 ]; then
    STEPS_THIS_RUN=2
    OUTPUT_LOG=log_compile-$NODE_ID.log
elif [ -v PERF_TEST ] && [ $PERF_TEST -gt 0 ]; then
    STEPS_THIS_RUN=100
    OUTPUT_LOG=log_exe-$NODE_ID.log
else
    STEPS_THIS_RUN=-1
    OUTPUT_LOG=log_exe-$NODE_ID.log
fi

echo NEURON_CC_FLAGS=$NEURON_CC_FLAGS
echo NUM_NEURONCORES=$NUM_NEURONCORES
echo NEURON_RT_NUM_CORES=$NEURON_RT_NUM_CORES
echo TP_DEGREE=$TP_DEGREE
echo USE_MIX_PRECISION=$USE_MIX_PRECISION
echo USE_ZERO_1=$USE_ZERO_1
echo GBS=$GBS
echo MBS=$MBS
echo TOTAL_STEPS=$TOTAL_STEPS
echo WARMUP_STEPS=$WARMUP_STEPS
echo LR=$LR
echo MODEL_PATH=$MODEL_PATH
echo IGNORE_INDEX=$IGNORE_INDEX
echo DATA_PATH=$DATA_PATH
echo DO_EVAL=$DO_EVAL
echo EVAL_DATA_PATH=$EVAL_DATA_PATH
echo EVAL_STEPS=$EVAL_STEPS
echo SEQ_LEN=$SEQ_LEN

echo EXTRA_ARGS=$EXTRA_ARGS
echo DP=$DP
echo ACC_STEPS=$ACC_STEPS
echo STEPS_THIS_RUN=$STEPS_THIS_RUN
echo OUTPUT_LOG=$OUTPUT_LOG

echo "CHECKPOINTING AND LOADING FROM CHECKPOINT/PRETRAINED"
### Checkpointing
echo CHECKPOINT_DIR=$CHECKPOINT_DIR
echo CHECKPOINT_FREQ=$CHECKPOINT_FREQ

echo LOADING_STEP=$LOADING_STEP
echo PRETRAINED_WEIGHT=$PRETRAINED_WEIGHT


torchrun $DISTRIBUTED_ARGS \
    tp_zero1_dsk_run_eval.py \
    --model_path $MODEL_PATH \
    --data_dir $DATA_PATH \
    --do_eval $DO_EVAL \
    --eval_data_dir $EVAL_DATA_PATH \
    --eval_steps $EVAL_STEPS \
    --tensor_parallel_size $TP_DEGREE \
    --batch_size $MBS \
    --steps_this_run $STEPS_THIS_RUN\
    --max_steps $TOTAL_STEPS \
    --warmup_steps $WARMUP_STEPS \
    --lr $LR \
    --ignore_index $IGNORE_INDEX \
    --grad_accum_usteps $ACC_STEPS \
    --seq_len $SEQ_LEN \
    --sequence_parallel_enabled \
    --selective_checkpoint_enabled \
    --logging_interval 10 \
    --checkpoint_dir $CHECKPOINT_DIR \
    --checkpoint_freq $CHECKPOINT_FREQ \
    --loading_step $LOADING_STEP \
    $EXTRA_ARGS |& tee $OUTPUT_LOG
exit ${PIPESTATUS[0]}
