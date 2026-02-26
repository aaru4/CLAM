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
