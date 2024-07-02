# run model training with optimum[neuron] on the AWS Traininum instance


from dataclasses import dataclass, field
from typing import Optional
import logging
import sys
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from transformers.trainer_utils import get_last_checkpoint
from optimum.neuron.distributed import lazy_load_for_parallelism
from optimum.neuron import NeuronTrainingArguments as TrainingArguments
from optimum.neuron import NeuronHfArgumentParser as HfArgumentParser
from optimum.neuron import NeuronTrainer as Trainer

from pickle_dataset import PickledTrainerDataset

import os
# import transformers
# transformers.utils.logging.set_verbosity_debug()

@dataclass
class ScriptArgs:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    dataset_paths: list[str] = field(
        metadata={"help": "Paths where to look for tokenized data."}
    )
    validation_paths: Optional[list[str]] = field(
        default=None,
        metadata={"help": "Paths where to look for tokenized validation data."}
    )
    model_base: str = field(
        default='deepseek-ai/deepseek-coder-1.3b-base',
        # default='"deepseek-ai/deepseek-coder-6.7b-base',
        metadata={"help": "Model identifier from huggingface.co/models"}
    )
    shuffle: bool = field(
        default=True,
        metadata={"help": "Shuffle the dataset"}
    )
    pad_index: int = field(
        default=32018,
        metadata={"help": "Index of the padding token"}
    )

def main():
    parser = HfArgumentParser((ScriptArgs, TrainingArguments))
    print(f"{sys.argv=}")
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        print(f"Parsing script arguments from a JSON file: {os.path.abspath(sys.argv[1])}")
        script_args, train_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]), allow_extra_keys=True)
    elif len(sys.argv) == 2 and sys.argv[1].endswith(('yaml', 'yml')):
        print(f"Parsing script arguments from a YAML file: {os.path.abspath(sys.argv[1])}")
        script_args, train_args = parser.parse_yaml_file(yaml_file=os.path.abspath(sys.argv[1]), allow_extra_keys=True)
    else:
        print(f"Parsing script arguments from console")
        script_args, train_args, unknown_args = parser.parse_args_into_dataclasses(return_remaining_strings=True)
        if unknown_args:
            print(f"WARNING unknown_args: {unknown_args}")

    print(f"script_args: {script_args}")
    print(f"train_args: {train_args}")

    # load the data
    training_data = PickledTrainerDataset(script_args.dataset_paths)

    if train_args.do_eval:
        validation_data = PickledTrainerDataset(script_args.validation_paths)
    else:
        print("No evaluation in this run, not loading validation data")
        validation_data = None

    # Print device:
    import torch_xla.core.xla_model as xm
    print(f"Using device: {xm.xla_device()=}")

    # Manage compiler flags
    if 'NEURON_CC_FLAGS' in os.environ:
        NEURON_CC_FLAGS = os.environ['NEURON_CC_FLAGS'].split(" ")
        print(f"Pre-set compiler flags (NEURON_CC_FLAGS): {NEURON_CC_FLAGS}")
    else:
        print(f"No pre-set complier flags! (NEURON_CC_FLAGS=[]")
        NEURON_CC_FLAGS = []    

    # remove experimental compilation which causes ERRORs
    NEURON_CC_FLAGS = [flag for flag in NEURON_CC_FLAGS if 'experimental' not in flag] 

    # add retry_failed_compilation if its not there
    if not any(['retry_failed_compilation' in f for f in NEURON_CC_FLAGS]):
        NEURON_CC_FLAGS.append('--retry_failed_compilation')
        
    # --optlevel 1
    if not any(['optlevel' in f or 'O1' in f for f in NEURON_CC_FLAGS]):
        NEURON_CC_FLAGS.append("--optlevel=1")

    # uncomment to increase compiler verbosity
    # NEURON_CC_FLAGS.append(" --verbose=info")
    
    os.environ['NEURON_CC_FLAGS'] = " ".join(NEURON_CC_FLAGS)
    print(f"Final NEURON_CC_FLAGS: {os.environ['NEURON_CC_FLAGS']}")

    print(f"MALLOC_ARENA_MAX={os.environ['MALLOC_ARENA_MAX'] if 'MALLOC_ARENA_MAX' in os.environ else 'not defined'}")

    if 'NEURON_PARALLEL_COMPILE_MAX_RETRIES' in os.environ:
        print(f"{os.environ['NEURON_PARALLEL_COMPILE_MAX_RETRIES']=}, setting to 1")
    os.environ['NEURON_PARALLEL_COMPILE_MAX_RETRIES'] = '1'

    
    
    # Detecting last checkpoint. - HF code still not working
    # last_checkpoint = None
    # if os.path.isdir(train_args.output_dir) and train_args.do_train and not train_args.overwrite_output_dir:
    #     last_checkpoint = get_last_checkpoint(train_args.output_dir)
    #     if last_checkpoint is None and len(os.listdir(train_args.output_dir)) > 0:
    #         raise ValueError(
    #             f"Output directory ({train_args.output_dir}) already exists and is not empty. "
    #             "Use --overwrite_output_dir to overcome."
    #         )
    #     elif last_checkpoint is not None and train_args.resume_from_checkpoint is None:
    #         print(
    #             f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
    #             "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
    #         )

    
    # Detecting last checkpoint. # MY CODE
    # if train_args.resume_from_checkpoint:
    #     # set checkpoint_dir
    #     if isinstance(train_args.resume_from_checkpoint, str):
    #         checkpoint_dir = train_args.resume_from_checkpoint
    #         print(f"resume_form_checkpoint path provided, using it to find latest checkpoint: {train_args.resume_from_checkpoint}")
    #     else:
    #         checkpoint_dir = train_args.output_dir
    #         print(f"resume_from_training is True, using output_dir to find last checkpoint: {train_args.output_dir}")
        
    #     if os.path.isdir(checkpoint_dir):
    #         last_checkpoint = get_last_checkpoint(checkpoint_dir)
    #         if last_checkpoint:
    #             print(f"Found checkpoint to start with: {last_checkpoint}")
    #             train_args.resume_from_checkpoint = last_checkpoint # REMOVE, only for test
    #         else:
    #             raise ValueError(
    #                 f"Could not find a valid checkpoint in the provided directory: {checkpoint_dir}. "
    #                 "Provide a valid checkpoint path or use resume_from_checkpoint=False"
    #             )
    #     else:
    #         raise ValueError(
    #             f"Provided checkpoint directory does not exist ({train_args.output_dir}). "
    #             "Can't find any checkpoints to resume training! Provide a valid checkpoint path or use resume_from_checkpoint=False"
    #         )
    # else:
    #     print(f"resume_from_checkpoint is False, not looking for last checkpoint")
    #     last_checkpoint = None

    # Consolidate sharded checkpoint files to single file when TP degree > 1
    # if (int(os.environ.get("RANK", -1)) == 0) and int(training_args.tensor_parallel_size) > 1:
    #     print("Converting sharded checkpoint to consolidated format")
    #     from optimum.neuron.distributed.checkpointing import (
    #         consolidate_model_parallel_checkpoints_to_unified_checkpoint,
    #     )
    #     from shutil import rmtree

    #     consolidate_model_parallel_checkpoints_to_unified_checkpoint(
    #         training_args.output_dir, training_args.output_dir, "pytorch"
    #     )
    #     rmtree(os.path.join(training_args.output_dir, "tensor_parallel_shards"))  # remove sharded checkpoint files

    last_checkpoint = None # for now disable checkpoints and figure it out

    # Load model
    with lazy_load_for_parallelism(
        tensor_parallel_size=train_args.tensor_parallel_size
        # pipeline_parallel_size=training_args.pipeline_parallel_size, # looks like note ready yet
    ):
        model = AutoModelForCausalLM.from_pretrained(script_args.model_base)

    # Train the model
    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=training_data,
        eval_dataset=validation_data,
        # compute_metrics=compute_metrics,
        tokenizer=None,
        data_collator=None
        # data_collator=default_data_collator,
    )


    # if train_args.do_train: # Training - HF code
    #     checkpoint = None
    #     if train_args.resume_from_checkpoint is not None:
    #         checkpoint = train_args.resume_from_checkpoint
    #     elif last_checkpoint is not None:
    #         checkpoint = last_checkpoint
    #     train_result = trainer.train(resume_from_checkpoint=checkpoint)
   
    # Training - my code
    if train_args.do_train:
        # last_checkpoint = '/home/ubuntu/dev/train_llm/tmp_training_output_1_3bcpcp/checkpoint-50'
        if last_checkpoint is not None:
            # print(f"Using resume_from_checkpoint={last_checkpoint}")
            # print(f"resume_from_checkpoint arg in trn:={train_args.resume_from_checkpoint}")
            train_result = trainer.train(resume_from_checkpoint=last_checkpoint)
        else:
            train_result = trainer.train()
        
        trainer.save_model()  # Saves the tokenizer too for easy upload
        trainer.save_state()



# only needed when using torch_xla instead of optimum.neuron?
def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()

if __name__ == '__main__':
    main()