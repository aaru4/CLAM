import torch
import torch.nn.functional as F
import numpy as np
from lifelines.utils import concordance_index

N_BINS = 4


def survival_ce_loss(logits, survival_bin, event, eps=1e-7):
    B = logits.shape[0]
    loss = torch.zeros(B, device=logits.device)
    for i in range(B):
        k  = int(survival_bin[i].item())
        ev = int(event[i].item())
        if k < 0 or k >= N_BINS:
            continue
        masked_logits = logits[i].clone()
        masked_logits[k+1:] = 0.0
        h = torch.sigmoid(masked_logits[:k+1])
        if ev == 1:
            l = -torch.log(h[k].clamp(eps, 1-eps))
            if k > 0:
                l = l - torch.log(1 - h[:k].clamp(eps, 1-eps)).sum()
        else:
            l = -torch.log(1 - h[:k+1].clamp(eps, 1-eps)).sum()
        loss[i] = l
    return loss.mean()


def survival_nll_loss(hazards, survival_bin, event, eps=1e-8):
    B = hazards.shape[0]
    loss = torch.zeros(B, device=hazards.device)
    for i in range(B):
        k  = int(survival_bin[i].item())
        ev = int(event[i].item())
        if k < 0 or k >= N_BINS:
            continue
        masked_logits = hazards[i].clone()
        masked_logits[k+1:] = 0.0
        h = torch.sigmoid(masked_logits)
        S = torch.cumprod(1 - h, dim=0)
        S_padded = torch.cat([torch.ones(1, device=hazards.device), S])
        if ev == 1:
            loss[i] = -torch.log(S_padded[k] - S_padded[k+1] + eps)
        else:
            loss[i] = -torch.log(S_padded[k+1] + eps)
    return loss.mean()
