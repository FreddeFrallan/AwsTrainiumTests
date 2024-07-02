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

from torch.nn import CrossEntropyLoss
# from optimum.neuron import NeuronTrainer as Trainer
from optimum.neuron import NeuronTrainer

import torch_xla.core.xla_model as xm

from pickle_dataset import PickledTrainerDataset

import os
import numpy as np
# import transformers
# transformers.utils.logging.set_verbosity_debug()



class DSKTrainer(NeuronTrainer):
    def __init__(self, *args, **kwargs):
        # print(f"DEBUG: Initializing DSKTrainer instance with {args=}, {kwargs=}")
        self._pad_index = kwargs.pop('pad_index', None)
        print(f"pad_index is: {self._pad_index}")
        
        super().__init__(*args, **kwargs)
        

    @staticmethod
    def _masked_loss(loss_func, logits, labels, vocab_size):
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        shift_logits = shift_logits.view(-1, vocab_size)
        shift_labels = shift_labels.view(-1)

        # calculate the loss
        loss = loss_func(shift_logits, shift_labels)
        return loss
       
    # def compute_loss(self, model, inputs, return_outputs=False):
    #     # from compute_loss from the optiumu.neuron
    #     # self.state.last_inputs = inputs
    #     # self.trigger_on_step_middle_for_neuron_cache_callback(model)
    #     # print(f"DEBUG WHAT IS WHERE: {model.device=}, {xm.xla_device()=} {[(k, v.device) for k, v in inputs.items()]}" )
    #     labels = inputs.get("labels")
    #     outputs = model(**inputs)
    #     # outputs = model(input_ids, labels=labels)
    #     logits = outputs.get("logits")
        
    #     # print(f"DEBUG: {logits.device=}") # DEBUG: logits.device=device(type='xla', index=0)
    #     # print(f"DEBUG: {labels.device=}") # DEBUG: labels.device=device(type='xla', index=0)
    #     # loss_fct = CrossEntropyLoss(ignore_index=self._pad_index) # here we ignore the impact the pad token.
    #     loss_fct = CrossEntropyLoss()
    #     # no need to move .to device next line
        
    #     loss = DSKTrainer._masked_loss(loss_fct, logits, labels, model.config.vocab_size)

    #     # print(f"DEBUG LOSS FOR ME PLEASE: {model.device=}, {type(loss)=} {loss.dtype=} {loss.shape} {loss.device}")
    #     return (loss, outputs) if return_outputs else loss
    
    
def init_compute_metrics(loss_fcn, vocab_size):
    def compute_metrics_fun(eval_preds):
        logits, labels = eval_preds
        print(f"DEBUG compute_metrics with {logits.shape=} {labels.shape=}")
        # predictions = np.argmax(logits, axis=-1)
        # loss_fcn = CrossEntropyLoss(ignore_index=self._pad_index) # this one can probably be moved 
        loss = DSKTrainer._masked_loss(loss_fcn, logits, labels, vocab_size)
        return {
            "mean_loss": loss.mean()
            # more can be added
        }
    
    print(f"Initialized compute_metrics function with {loss_fcn} and {vocab_size=}")
    return compute_metrics_fun

def compute_metrics_not_fun(eval_preds):
    print(f"DEBUG compute_metrics just called!")
    # return {
    #     "some_dumb_metric": 0
    # }
    def _masked_loss(loss_func, logits, labels, vocab_size):
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        shift_logits = shift_logits.view(-1, vocab_size)
        shift_labels = shift_labels.view(-1)

        # calculate the loss
        loss = loss_func(shift_logits, shift_labels)
        return loss
    print(f"DEBUG compute_metrics, entering with {type(eval_preds)=}, {len(eval_preds)=}")
    logits, labels = eval_preds
    print(f"DEBUG compute_metrics with {logits.shape=} {labels.shape=}")
    # predictions = np.argmax(logits, axis=-1)
    loss_fcn = CrossEntropyLoss(ignore_index=32018) 
    loss = _masked_loss(loss_fcn, logits, labels, 32256)    
    return {
        "mean_loss": loss.mean()
        # more can be added
    }


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

    # initialize compute_metrics function
    # loss_fcn = CrossEntropyLoss(ignore_index=script_args.pad_index)
    # compute_metrics = init_compute_metrics(loss_fcn, model.config.vocab_size)

    # Train the model
    # Initialize Trainer
    trainer = DSKTrainer(
        model=model,
        args=train_args,
        train_dataset=training_data,
        eval_dataset=validation_data,
        # compute_metrics=compute_metrics,
        # compute_metrics=compute_metrics_not_fun,
        tokenizer=None,
        data_collator=None,
        pad_index=script_args.pad_index
        # data_collator=default_data_collator,
    )

    for batch in trainer.get_eval_dataloader(validation_data):
        print("EVAL DATA DEBUG LOOP")
        print(batch)
        input_ids = batch.get("input_ids")
        labels = batch.get("labels")
        print(f"{input_ids.shape=}  {labels.shape=}")
        break


    # if train_args.do_train: # Training - HF code
    #     checkpoint = None
    #     if train_args.resume_from_checkpoint is not None:
    #         checkpoint = train_args.resume_from_checkpoint
    #     elif last_checkpoint is not None:
    #         checkpoint = last_checkpoint
    #     train_result = trainer.train(resume_from_checkpoint=checkpoint)
   
    # Training - my code
    if train_args.do_train:
        # last_checkpoint = '/home/ubuntu/dev/train_llm/tmp_training_output_1_3b/checkpoint-100'
        # last_checkpoint = '/home/ubuntu/dev/train_llm/tmp_training_output_1_3b/checkpoint-manual'
        last_checkpoint = None
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