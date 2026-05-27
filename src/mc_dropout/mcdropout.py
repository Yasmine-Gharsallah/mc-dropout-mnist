"""Core MC Dropout machinery: prediction in standard and MC modes, plus the
uncertainty measures derived from the stochastic forward passes.

Conventions
-----------
`probs_T` is a tensor of shape (T, N, C): T stochastic forward passes, N inputs,
C classes. All uncertainty functions take either `probs_T` or the mean
predictive distribution `mean_probs` of shape (N, C).
"""

import torch
import torch.nn.functional as F


def enable_dropout(model):
    """Put only the Dropout layers back into training mode.

    Everything else (e.g. any BatchNorm) stays in eval mode. Call this *after*
    `model.eval()` to perform MC Dropout.
    """
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.train()


@torch.no_grad()
def standard_predict(model, loader, device):
    """Deterministic prediction with dropout OFF.

    Returns (probs, labels) with probs of shape (N, C) and labels of shape (N,).
    """
    model.eval()
    probs, labels = [], []
    for x, y in loader:
        x = x.to(device)
        p = F.softmax(model(x), dim=1).cpu()
        probs.append(p)
        labels.append(y)
    return torch.cat(probs), torch.cat(labels)


@torch.no_grad()
def mc_forward(model, loader, device, T=30):
    """Run T stochastic forward passes with dropout ON.

    Returns (probs_T, labels) with probs_T of shape (T, N, C).
    """
    model.eval()
    enable_dropout(model)

    probs_T_batches, labels = [], []
    for x, y in loader:
        x = x.to(device)
        stack = torch.stack([F.softmax(model(x), dim=1) for _ in range(T)], dim=0)  # (T,B,C)
        probs_T_batches.append(stack.cpu())
        labels.append(y)
    probs_T = torch.cat(probs_T_batches, dim=1)  # (T, N, C)
    return probs_T, torch.cat(labels)


@torch.no_grad()
def mc_forward_batch(model, x, T=30):
    """Run T stochastic passes on a single batch `x` (already on the right device).

    Returns probs_T of shape (T, B, C). Assumes dropout is already enabled.
    """
    return torch.stack([F.softmax(model(x), dim=1) for _ in range(T)], dim=0)


# ---------------------------------------------------------------------------
# Uncertainty measures
# ---------------------------------------------------------------------------

def predictive_mean(probs_T):
    """Mean predictive distribution, shape (N, C)."""
    return probs_T.mean(dim=0)


def predictive_entropy(mean_probs, eps=1e-12):
    """Total predictive uncertainty: H[ mean_p ] = -sum_c p_c log p_c. Shape (N,)."""
    return -(mean_probs * torch.log(mean_probs + eps)).sum(dim=1)


def expected_entropy(probs_T, eps=1e-12):
    """Aleatoric part: average over the T passes of each pass's entropy. Shape (N,)."""
    ent_per_pass = -(probs_T * torch.log(probs_T + eps)).sum(dim=2)  # (T, N)
    return ent_per_pass.mean(dim=0)


def mutual_information(probs_T, eps=1e-12):
    """Epistemic (model) uncertainty, a.k.a. BALD:
    MI = H[ mean_p ] - E_t[ H[p_t] ]. Shape (N,).
    """
    mean_probs = predictive_mean(probs_T)
    return predictive_entropy(mean_probs, eps) - expected_entropy(probs_T, eps)


def predictive_variance(probs_T):
    """Variance across the T passes, averaged over classes. Shape (N,)."""
    return probs_T.var(dim=0).mean(dim=1)


def negative_log_likelihood(mean_probs, labels, eps=1e-12):
    """Per-example NLL of the true class under the mean predictive distribution. Shape (N,)."""
    p_true = mean_probs[torch.arange(mean_probs.size(0)), labels]
    return -torch.log(p_true + eps)


def summarise(probs_T, labels):
    """Convenience bundle of predictions and all uncertainty measures."""
    mean_probs = predictive_mean(probs_T)
    preds = mean_probs.argmax(dim=1)
    return {
        "mean_probs": mean_probs,
        "preds": preds,
        "entropy": predictive_entropy(mean_probs),
        "variance": predictive_variance(probs_T),
        "mutual_information": mutual_information(probs_T),
        "nll": negative_log_likelihood(mean_probs, labels),
    }
