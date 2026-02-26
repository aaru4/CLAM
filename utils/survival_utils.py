import torch
import torch.nn.functional as F

N_BINS = 4


def survival_ce_loss(logits, survival_bin, event, eps=1e-7):
    B = logits.shape[0]
    loss = torch.zeros(B, device=logits.device)

    for i in range(B):
        k  = int(survival_bin[i].item())
        ev = int(event[i].item())

        if k < 0 or k >= N_BINS:
            continue

        # Mask: only bins 0..k are active, bins after k set to 0
        masked_logits = logits[i].clone()
        masked_logits[k+1:] = 0.0                      # mask out bins after k

        h = torch.sigmoid(masked_logits[:k+1])          # [k+1] hazard probs

        if ev == 1:
            # event: target=1 at bin k, target=0 for bins before k
            # -log(h_k) - sum_{j<k} log(1 - h_j)
            l = -torch.log(h[k].clamp(eps, 1-eps))
            if k > 0:
                l = l - torch.log(1 - h[:k].clamp(eps, 1-eps)).sum()
        else:
            # censored: target=0 for all bins up to and including k
            # -sum_{j<=k} log(1 - h_j)
            l = -torch.log(1 - h[:k+1].clamp(eps, 1-eps)).sum()

        loss[i] = l

    return loss.mean()

def survival_nll_loss(hazards, survival_bin, event, eps=1e-8):
    """

    Args:
        hazards      : Tensor [B, N_BINS]  raw logits
        survival_bin : Tensor [B]          0-indexed bin
        event        : Tensor [B]          1=progressed, 0=censored
    Returns:
        scalar mean loss
    """
    B = hazards.shape[0]
    loss = torch.zeros(B, device=hazards.device)

    for i in range(B):
        k  = int(survival_bin[i].item())
        ev = int(event[i].item())

        if k < 0 or k >= N_BINS:
            continue

        # Same masking as CE: zero out logits after bin k
        masked_logits = hazards[i].clone()
        masked_logits[k+1:] = 0.0

        # Hazard per bin, survival function, pad with S(t=0)=1
        h = torch.sigmoid(masked_logits)                          # [N_BINS]
        S = torch.cumprod(1 - h, dim=0)                          # [N_BINS]
        S_padded = torch.cat([torch.ones(1, device=hazards.device), S])  # [N_BINS+1]

        if ev == 1:
            # Event occurred at bin k: -log(S(k-1) - S(k))
            loss[i] = -torch.log(S_padded[k] - S_padded[k+1] + eps)
        else:
            # Censored at bin k: patient survived at least to bin k
            # -log(S(k)) — probability of surviving up to and including bin k
            loss[i] = -torch.log(S_padded[k+1] + eps)

    return loss.mean()
