import json
import math
import os
import queue
import time
from datetime import datetime, timezone
from functools import partial
from itertools import chain
from typing import Any, Dict, List

import datasets
import torch
from torch.utils.data import DistributedSampler
from torch.utils.data.dataloader import DataLoader
from transformers import default_data_collator, set_seed

from pickle_dataset import PickledTrainerDataset

try:
    from lr import CosineAnnealing
except ImportError:
    CosineAnnealing = None

from collections import namedtuple

Metric = namedtuple("Metric", ["name", "value", "units", "additional_data"])
remainder = {"input_ids": [], "attention_mask": [], "token_type_ids": []}


def get_learning_rate_scheduler(optimizer, args, last_epoch=-1):
    lr_scheduler = CosineAnnealing(
        optimizer,
        max_steps=args.max_steps,
        min_lr=args.min_lr,
        warmup_steps=args.warmup_steps,
        constant_steps=args.constant_steps,
        last_epoch=last_epoch,
    )
    return lr_scheduler


def get_param_groups_by_weight_decay(model):
    """Get param groups."""
    if hasattr(model, "local_named_parameters"):
        # Zero1 use the first param in opt to decide the device
        param_optimizer = list(model.local_named_parameters())
    else:
        param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm"]  # gamma/beta are in LayerNorm.weight

    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
            "weight_decay": 0.01,
        },
        {
            "params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    return optimizer_grouped_parameters


def create_dsk_pretraining_dataset(data_dir, mini_batch_size, dp_size, dp_rank, seed):
    # Workaround because python functions are not picklable
    class WorkerInitObj(object):
        def __init__(self, seed):
            self.seed = seed

        def __call__(self, id):
            set_seed(self.seed)

    worker_init = WorkerInitObj(seed)
    # train_data = datasets.load_from_disk(data_dir)
    train_data = PickledTrainerDataset([data_dir])

    train_sampler = DistributedSampler(
        train_data,
        num_replicas=dp_size,
        rank=dp_rank,
        shuffle=False,
        drop_last=True,
    )
    train_dataloader = DataLoader(
        train_data,
        collate_fn=default_data_collator,
        sampler=train_sampler,
        batch_size=mini_batch_size,
        num_workers=0,
        worker_init_fn=worker_init,
        drop_last=True,
        pin_memory=True,
    )
    return train_dataloader, None

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


def print_logs(loss, global_norm, args, throughput, logger, total_steps, current_lr, input_ids, start):
    total_norm_cpu = global_norm.cpu().item()
    logger.log(total_steps, loss, total_norm_cpu, current_lr, input_ids, throughput, start)


class TrainingMetrics:
    """
    This class is used for logging metrics to a json file. One can provide a
    dictionary of metrics that needs to be stored, and it wpuld get
    written to the file.
    Arguments:
        json_file: File used for logging. If no file exists, new file would be created.
    """

    def __init__(self, json_file):
        self.json_file = json_file

    def read_modify_write_file(self, data, key: str = "metrics") -> None:
        """
        data (dict of training parameters or list of metrics): Data to update in the file.
        key (str): the dictionary key under which data is to be recorded
        """
        result_dict = {}
        print(f"Writing data to the provided results file: {self.json_file}")
        if os.path.exists(self.json_file):
            with open(self.json_file, "r") as json_file:
                content = json_file.read()
                if not content.strip():  # Check if content is empty or contains only whitespace
                    print("File is empty or contains only whitespace.")
                else:
                    result_dict = json.loads(content) or result_dict
        print(f"Updating with {key} data: {data}")
        if result_dict:
            try:
                # handle internal named entity if present
                results = result_dict[next(iter(result_dict))]
            except Exception:
                results = result_dict
            current = results.get(key)
            if not current:
                results[key] = data
            else:
                if isinstance(current, list):
                    current.extend(data)
                elif isinstance(current, dict):
                    current.update(data)
        else:
            result_dict["results"] = {key: data}
        with open(self.json_file, "w") as json_file:
            json.dump(result_dict, json_file)

    def store_metrics(self, metrics: List[Metric]) -> None:
        """
        Writes collected metrics to the file.
        """
        data = [
            {
                "MetricName": metric.name,
                "MeasuredValue": metric.value,
                "Units": metric.units,
                "Timestamp": datetime.now(timezone.utc).isoformat(),
                "AdditionalData": metric.additional_data,
            }
            for metric in metrics
        ]
        self.update(data=data, key="metrics")

    def store_parameters(self, parameters: Dict[str, Any]) -> None:
        """
        Writes specified model and configuration parameters to the file.
        """
        self.update(data=parameters, key="parameters")

    def update(self, **kwargs: Any) -> None:
        """
        Write specified data to the output file.
        """
        self.read_modify_write_file(**kwargs)


class MovingAvgWindowMetirc:
    def __init__(self, moving_avg_window_size=10, logging_interval=1):
        self.moving_avg_window_size = math.ceil(moving_avg_window_size / logging_interval)
        self.moving_avg_window = queue.Queue()
        self.window_time = 0
        self.start_time = time.time()

    def get_value(self):
        step_time = time.time() - self.start_time
        self.start_time += step_time
        self.window_time += step_time
        self.moving_avg_window.put(step_time)
        window_size = self.moving_avg_window.qsize()
        if window_size > self.moving_avg_window_size:
            self.window_time -= self.moving_avg_window.get()
            window_size -= 1
        return self.compute_metric(window_size)
        

    def compute_metric(self, window_size):
        '''Compute metric value based on class specific data and average window
        '''
        pass

class Throughput(MovingAvgWindowMetirc):
    def __init__(self, batch_size, world_size, grad_accum_usteps, moving_avg_window_size=10, logging_interval=1):
        super().__init__(moving_avg_window_size, logging_interval)
        self.seqs_per_iteration = batch_size * world_size * grad_accum_usteps * logging_interval

    def compute_metric(self, window_size):
        return window_size * self.seqs_per_iteration / self.window_time
    
class MFU(MovingAvgWindowMetirc):
    def __init__(self, batch_size, world_size, grad_accum_usteps, seq_length, model_FLOP, nc_FLOPS, no_neuron_cores, moving_avg_window_size=10, logging_interval=1):
        '''Computes Model FLOPs utilization (MFU): https://arxiv.org/pdf/2204.02311 (see Appendix B)
        This metric tells what is the utilization of the hardware FLOPS as a ratio of the maximum FLOPS
        '''
        super().__init__(moving_avg_window_size, logging_interval)
        self.tokens_per_iteration = batch_size * world_size * grad_accum_usteps * logging_interval * seq_length
        self.model_FLOP = model_FLOP
        self.hw_FLOPS = no_neuron_cores * nc_FLOPS

    def compute_metric(self, window_size):
        tokens_per_second = window_size * self.tokens_per_iteration / self.window_time
        return self.model_FLOP * tokens_per_second / self.hw_FLOPS


# class Throughput:
#     def __init__(self, batch_size, world_size, grad_accum_usteps, moving_avg_window_size=10, logging_interval=1):
#         """
#         Used to calculate the throughput over a moving window. It records the step time
#         between two calls and uses that time to calculate the throughput.
#         """
#         self.seqs_per_iteration = batch_size * world_size * grad_accum_usteps * logging_interval
#         self.moving_avg_window_size = math.ceil(moving_avg_window_size / logging_interval)
#         self.moving_avg_window = queue.Queue()
#         self.window_time = 0
#         self.start_time = time.time()

#     def get_throughput(self):
#         step_time = time.time() - self.start_time
#         self.start_time += step_time
#         self.window_time += step_time
#         self.moving_avg_window.put(step_time)
#         window_size = self.moving_avg_window.qsize()
#         if window_size > self.moving_avg_window_size:
#             self.window_time -= self.moving_avg_window.get()
#             window_size -= 1
#         throughput = window_size * self.seqs_per_iteration / self.window_time
#         return throughput
