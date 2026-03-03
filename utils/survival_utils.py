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

def cox_ph_loss(risk, time_to_event, event):
    """
    Cox proportional hazards negative partial log-likelihood 

    Inputs
    ------
    risk : Tensor of shape (N,) or (N, 1)
        Model output "risk score" per patient/bag. In Cox, this is the *linear predictor* (i.e. don't use softmax):
        IMPORTANT: these are NOT probabilities. Higher risk => higher hazard (earlier events).

    time_to_event : Tensor of shape (N,)
        Observed follow-up time (event time if event==1, censor time if event==0).

    event : Tensor of shape (N,)
        Event indicator: 1 if event observed, 0 if censored.

    Returns
    -------
    loss : Tensor (scalar)
        Negative partial log-likelihood averaged over the number of observed events.
    """

    # Flatten to a 1D vector of length N (one score per patient/bag)
    risk = risk.view(-1)

    # Sort subjects by descending time.
    # After sorting this way, for an index i, all indices <= i correspond to
    # subjects with time >= time[i] (i.e., those still "at risk" at time[i]).
    order = torch.argsort(time_to_event.view(-1), descending=True)
    risk = risk[order]
    event = event.view(-1)[order].float()

    # Cox partial likelihood for each subject i with an observed event:
    #   log( exp(risk_i) / sum_{j in R_i} exp(risk_j) )
    # = risk_i - log( sum_{j in R_i} exp(risk_j) )
    #
    # Here, R_i is the "risk set" at time_i (everyone who has not yet had an event
    # or been censored before time_i).
    #
    # Because we sorted by descending time, the risk set sum_{j in R_i} exp(risk_j)
    # becomes a cumulative sum over exp(risk) from 0..i.
    #
    # logcumsumexp computes:
    #   log( exp(risk_0) + exp(risk_1) + ... + exp(risk_i) )
    # in a numerically stable way (avoids overflow/underflow).
    log_cumsum_risk = torch.logcumsumexp(risk, dim=0)

    # For each i, this is:
    #   risk_i - log(sum_{j in R_i} exp(risk_j))
    # which is the log partial-likelihood contribution for subject i.
    partial = risk - log_cumsum_risk

    # Only observed events contribute to the Cox partial likelihood.
    # Censored subjects define the risk sets, but do not add a numerator term.
    #
    # We average by the number of observed events to keep the loss scale
    # more consistent across batches.
    denom = event.sum().clamp(min=1.0)

    # Negative sign because we minimize loss (maximize log-likelihood).
    # Multiply by event to include only event cases.
    return -(partial * event).sum() / denom
