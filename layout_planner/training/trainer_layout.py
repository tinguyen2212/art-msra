import numpy as np
from queue import Queue
from collections import defaultdict

import torch
from transformers import Trainer

from .utils import accuracy


def l2_loss(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Args:
        u: (N, T, D) tensor.
        v: (N, T, D) tensor.
    Returns:
        l1_loss: (N,) tensor of summed L1 loss.
    """
    assert u.shape == v.shape, (u.shape, v.shape)
    return ((u - v) ** 2).sum(dim=-1) ** 0.5


def batch_purity(input_ids, tokenizer):
    strings = tokenizer.batch_decode(input_ids)
    strings = [
        string.replace("<s> ", "").replace('> "', '>"').replace("</s>", "")
        for string in strings
    ]
    return strings


class Meter:
    def __init__(self,size):
        self.size = size
        self.reset()
    
    def reset(self):
        self.bins = defaultdict(Queue)

    def update(self,metrics):
        for k,v in metrics.items():
            self.bins[k].put(v)
    
    def get(self):
        metrics = {}
        for k,v in self.bins.items():
            metrics[k] = np.mean(list(v.queue))
        return metrics


class TrainerLayout(Trainer):
    def __init__(self, extra_args, **kwargs):
        self.quantizer = kwargs.pop("quantizer", None)
        super().__init__(**kwargs)
        self.extra_args = extra_args
        weight = torch.ones(self.extra_args.vocab_size)
        weight[self.extra_args.old_vocab_size :] = self.extra_args.new_token_weight
        self.weighted_ce_loss = torch.nn.CrossEntropyLoss(weight=weight).cuda()
        if 'opt-' in self.extra_args.opt_version:
            if self.args.fp16:
                self.weighted_ce_loss = self.weighted_ce_loss.half()
            elif self.args.bf16:
                self.weighted_ce_loss = self.weighted_ce_loss.bfloat16()

        self.meter = Meter(self.args.logging_steps)

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels").long()
        
        (
            model_output,
            full_labels,
            input_embs_norm,
        ) = model(labels=labels)
        
        output = model_output.logits

        masked_full_labels = full_labels.clone()
        masked_full_labels[masked_full_labels < self.extra_args.old_vocab_size] = -100

        weighted_ce_loss = self.weighted_ce_loss(
            output[:, :-1, :].reshape(-1, output.shape[-1]),
            full_labels[:, 1:].reshape(-1),
        )
        ce_loss = weighted_ce_loss * self.extra_args.ce_loss_scale

        loss = ce_loss
        acc1, acc5 = accuracy(output[:, :-1, :], full_labels[:, 1:], -100, topk=(1, 5))
        masked_acc1, masked_acc5 = accuracy(
            output[:, :-1, :], masked_full_labels[:, 1:], -100, topk=(1, 5)
        )

        metrics = {
            "loss": loss.item(),
            "ce_loss": ce_loss.item(),
            "top1": float(acc1),
            "top5": float(acc5),
            "masked_top1": float(masked_acc1),
            "masked_top5": float(masked_acc5),
            "inp_emb_norm": input_embs_norm.item(),
        }
        self.meter.update(metrics)
        if self.state.global_step % self.args.logging_steps == 0:
            metrics = self.meter.get()
            self.meter.reset()
            self.log(metrics)

        outputs = model_output

        return (loss, outputs) if return_outputs else loss
