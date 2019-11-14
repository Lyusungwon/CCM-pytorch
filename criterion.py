import torch
import torch.nn as nn
import torch.nn.functional as F

from dataloader import PAD_IDX
        

def criterion(output, pointer_prob, target, pointer_prob_target):
    # output: logits
    nll_loss = F.cross_entropy(output, target, ignore_index=PAD_IDX, reduction='mean')
    pointer_prob_loss = F.binary_cross_entropy(pointer_prob, pointer_prob_target, reduction='mean')
    return nll_loss + pointer_prob_loss, nll_loss

def perplexity(nll_loss):
    return torch.exp(nll_loss).mean()
    