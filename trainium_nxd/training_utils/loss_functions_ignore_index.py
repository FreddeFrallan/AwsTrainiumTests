# NOTE:
# in order to use it, you need to put it into
# neuronx_distributed.parallel_layers
# and add it to this init file:
# <python packages>site-packages/neuronx_distributed/parallel_layers/__init__.py

import torch

from .parallel_state import (
    get_tensor_model_parallel_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_size,
)
from .utils import EmbeddingUtility


class _ParallelCrossEntropy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, vocab_parallel_logits, target, ignore_index=-100):
        pass

        # Maximum value along vocab dimension across all GPUs.
        logits_max = torch.max(vocab_parallel_logits, dim=-1)[0]
        torch.distributed.all_reduce(
            logits_max,
            op=torch.distributed.ReduceOp.MAX,
            group=get_tensor_model_parallel_group(),
        )
        # Subtract the maximum value.
        vocab_parallel_logits.sub_(logits_max.unsqueeze(dim=-1))

        # Get the partition's vocab indecies
        get_vocab_range = EmbeddingUtility.range_from_per_partition_vocab_size
        partition_vocab_size = vocab_parallel_logits.size()[-1]
        rank = get_tensor_model_parallel_rank()
        world_size = get_tensor_model_parallel_size()
        vocab_start_index, vocab_end_index = get_vocab_range(partition_vocab_size, rank, world_size)

        # Create a mask of valid vocab ids (1 means it needs to be masked).
        # masked_target = target.clone() - vocab_start_index
        # masked_target[target_mask] = 0
        # new xla friendly
        target_mask = (target >= vocab_start_index) & (target < vocab_end_index) & (target != ignore_index)
        masked_target = target.clone() - vocab_start_index
        masked_target = torch.mul(masked_target, target_mask.long())

        # Get predicted-logits = logits[target].
        # For Simplicity, we convert logits to a 2-D tensor with size
        # [*, partition-vocab-size] and target to a 1-D tensor of size [*].
        logits_2d = vocab_parallel_logits.view(-1, partition_vocab_size)
        masked_target_1d = masked_target.view(-1)
        arange_1d = torch.arange(start=0, end=logits_2d.size()[0], device=logits_2d.device, dtype=torch.long)
        predicted_logits_1d = logits_2d[arange_1d, masked_target_1d]
        predicted_logits_1d = predicted_logits_1d.clone().contiguous()
        predicted_logits = predicted_logits_1d.view_as(target)
        # original
        # predicted_logits[target_mask] = 0.0
        # new xla friendly
        predicted_logits = torch.mul(predicted_logits, target_mask.float())
        # All reduce is needed to get the chunks from other GPUs.
        torch.distributed.all_reduce(
            predicted_logits,
            op=torch.distributed.ReduceOp.SUM,
            group=get_tensor_model_parallel_group(),
        )

        # Sum of exponential of logits along vocab dimension across all GPUs.
        # new xla friendly
        # exp_logits = vocab_parallel_logits
        # torch.exp(vocab_parallel_logits, out=exp_logits)
        exp_logits = torch.exp(vocab_parallel_logits)

        # Ignore ignore_index for
        mask_ignore_index = (target != ignore_index).float()
        exp_logits.mul_(mask_ignore_index.view(-1, 1))

        sum_exp_logits = exp_logits.sum(dim=-1)
        torch.distributed.all_reduce(
            sum_exp_logits,
            op=torch.distributed.ReduceOp.SUM,
            group=get_tensor_model_parallel_group(),
        )

        # Loss = log(sum(exp(logits))) - predicted-logit.
        loss = torch.log(sum_exp_logits) - predicted_logits
        loss.mul_(mask_ignore_index)

        # Store softmax, target-mask and masked-target for backward pass.
        exp_logits.div_(sum_exp_logits.unsqueeze(dim=-1) + 1e-8)  # to prevent division by 0 for ignore_index

        vocab_size = exp_logits.size(-1)

        ctx.ignore_index, ctx.vocab_size = ignore_index, vocab_size
        ctx.save_for_backward(exp_logits, target_mask, masked_target_1d)

        return loss

    @staticmethod
    def backward(ctx, grad_output):
        # Retreive tensors from the forward path.
        softmax, target_mask, masked_target_1d = ctx.saved_tensors
        ignore_index, vocab_size = ctx.ignore_index, ctx.vocab_size

        # All the inputs have softmax as thier gradient.
        grad_input = softmax
        # For simplicity, work with the 2D gradient.
        partition_vocab_size = softmax.size()[-1]
        grad_2d = grad_input.view(-1, partition_vocab_size)

        # Add the gradient from matching classes.
        arange_1d = torch.arange(start=0, end=grad_2d.size()[0], device=grad_2d.device, dtype=torch.long)
        grad_2d[arange_1d, masked_target_1d] -= target_mask.view(-1).float()

        # Finally elementwise multiplication with the output gradients.
        grad_input.mul_(grad_output.unsqueeze(dim=-1))

        return grad_input, None, None


def parallel_cross_entropy(vocab_parallel_logits, target, ignore_index=-100):
    """Helper function for the cross entropy."""
    # print(f"DEBUG APPLY IGNORE: {vocab_parallel_logits =}, {target=}, {ignore_index=}")
    return _ParallelCrossEntropy.apply(vocab_parallel_logits, target, ignore_index)
