import torch
import torch.nn as nn
import torch.nn.functional as F

from dataloader import PAD_IDX
        

def criterion(output, pointer_prob, target, pointer_prob_target):
    batch_size, rl = target.size()
    output_len = output.size()[2]
    if output_len > rl:
        output = output[:, :, :rl]
        pointer_prob = pointer_prob[:, :rl]
    elif output_len < rl:
        pad = torch.zeros((batch_size, output.size()[1], rl), device=output.device)
        pad[:, :, :output_len] = output
        output = pad
        pad = torch.zeros((batch_size, rl), device=output.device)
        pad[:, :output_len] = pointer_prob
        pointer_prob = pad

    # output: logits
    nll_loss = F.cross_entropy(output, target, ignore_index=PAD_IDX, reduction='mean')
    pointer_prob_loss = F.binary_cross_entropy(pointer_prob, pointer_prob_target, reduction='mean')
    return nll_loss + pointer_prob_loss, nll_loss


def perplexity(nll_loss):
    return torch.exp(nll_loss).mean()


def baseline_criterion(output, target):
    return F.cross_entropy(output, target, ignore_index=PAD_IDX, reduction='mean')

