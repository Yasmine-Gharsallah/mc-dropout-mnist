"""UCI-style regression: a small MLP with dropout, trained as in section 5.3
of Gal & Ghahramani (2016). The MC Dropout predictive log-likelihood uses
equation (8) of the paper:

    log p(y|x, X, Y) ~= logsumexp[-1/2 * tau * ||y - y_t||^2]
                        - log T - 0.5 log(2 pi) - 0.5 log(tau^-1)

`tau` is the model precision; the paper chooses it via Bayesian optimisation
over validation log-likelihood. Here we use a small grid search (still a
validation-based choice, just cheaper).
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


class DropoutMLP(nn.Module):
    """One-hidden-layer MLP with dropout before the hidden activation and
    before the output, mirroring the small model used in the paper's UCI
    regression experiments (50 hidden units)."""

    def __init__(self, in_dim: int, hidden: int = 50, p: float = 0.05):
        super().__init__()
        self.drop_in = nn.Dropout(p)
        self.fc1 = nn.Linear(in_dim, hidden)
        self.drop_h = nn.Dropout(p)
        self.fc2 = nn.Linear(hidden, 1)

    def forward(self, x):
        x = self.drop_in(x)
        x = F.relu(self.fc1(x))
        x = self.drop_h(x)
        return self.fc2(x).squeeze(-1)


def train_mlp(model, X_train, y_train, epochs=400, batch_size=32, lr=1e-3,
              weight_decay=1e-4, device="cpu"):
    """Standard regression training: Adam + MSE. Returns the trained model."""
    ds = TensorDataset(torch.as_tensor(X_train, dtype=torch.float32),
                       torch.as_tensor(y_train, dtype=torch.float32))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
    optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()
    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optim.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optim.step()
    return model


@torch.no_grad()
def mc_predict(model, X, T=1000, device="cpu"):
    """T stochastic passes with dropout ON. Returns array of shape (T, N)."""
    # Match the MC-Dropout convention used elsewhere: eval mode + dropout layers
    # re-enabled, so any future BN/etc. stays deterministic.
    model.eval()
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()
    Xt = torch.as_tensor(X, dtype=torch.float32, device=device)
    preds = torch.stack([model(Xt) for _ in range(T)], dim=0)  # (T, N)
    return preds.cpu().numpy()


def mc_metrics(preds_T, y_true_std, y_mean, y_std, tau):
    """Compute RMSE and predictive log-likelihood from MC samples.

    preds_T   : (T, N) standardised predictions
    y_true_std: (N,) standardised ground truth
    y_mean    : training-set mean of y (for un-standardising)
    y_std     : training-set std  of y
    tau       : model precision

    Returns (rmse_orig_scale, mean_test_log_likelihood).
    """
    # Un-standardise back to the original scale for RMSE & log-likelihood
    preds = preds_T * y_std + y_mean              # (T, N) original scale
    y_true = y_true_std * y_std + y_mean          # (N,) original scale
    mean_pred = preds.mean(axis=0)                # (N,)
    rmse = float(np.sqrt(np.mean((mean_pred - y_true) ** 2)))

    # Predictive log-likelihood, eq. (8) of the paper (per data point), in nats.
    T = preds.shape[0]
    sq = (preds - y_true[None, :]) ** 2           # (T, N)
    a = -0.5 * tau * sq                           # (T, N)
    # log-sum-exp over the T passes, then subtract log T to get the mean over t.
    lse = np.log(np.sum(np.exp(a - a.max(axis=0)), axis=0) + 1e-300) + a.max(axis=0)
    ll_per_point = lse - math.log(T) - 0.5 * math.log(2 * math.pi) - 0.5 * math.log(1.0 / tau)
    return rmse, float(ll_per_point.mean())


def search_tau(preds_T, y_true_std, y_mean, y_std,
               tau_grid=(0.05, 0.1, 0.15, 0.2, 0.25, 0.5, 1.0, 2.0)):
    """Pick tau that maximises validation log-likelihood."""
    best_tau, best_ll = None, -np.inf
    for tau in tau_grid:
        _, ll = mc_metrics(preds_T, y_true_std, y_mean, y_std, tau)
        if ll > best_ll:
            best_ll, best_tau = ll, tau
    return best_tau
