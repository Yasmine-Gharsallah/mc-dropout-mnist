"""Experiment 4 — out-of-distribution (OOD) detection with MC Dropout.

The model is trained only on MNIST. We compare its predictive uncertainty on
in-distribution MNIST test images against FashionMNIST images (same shape, never
seen in training). If MC Dropout uncertainty is meaningful, OOD inputs should get
higher entropy, and entropy alone should separate the two sets (high AUROC).

Produces:
  figures/ood_entropy_hist.png
  results/ood_summary.csv
"""

import argparse

import common  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from mc_dropout import (
    set_seed, get_device, MCDropoutCNN, get_mnist_loaders, get_ood_loader,
    mc_forward, predictive_mean, predictive_entropy, mutual_information,
)


def load_model(device):
    ckpt = torch.load(common.MODEL_PATH, map_location=device)
    p = ckpt.get("meta", {}).get("dropout", 0.5)
    model = MCDropoutCNN(p=p).to(device)
    model.load_state_dict(ckpt["state_dict"])
    return model


def uncertainty_scores(model, loader, device, T):
    probs_T, _ = mc_forward(model, loader, device, T=T)
    mean_probs = predictive_mean(probs_T)
    entropy = predictive_entropy(mean_probs).numpy()
    mi = mutual_information(probs_T).numpy()
    return entropy, mi


def main():
    parser = argparse.ArgumentParser(description="OOD detection with MC Dropout")
    parser.add_argument("--T", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device(args.device)
    print("Using device:", device)

    _, test_loader = get_mnist_loaders(batch_size=128, data_dir=common.DATA_DIR)
    ood_loader = get_ood_loader(batch_size=128, data_dir=common.DATA_DIR)
    model = load_model(device)

    ent_in, mi_in = uncertainty_scores(model, test_loader, device, args.T)
    ent_ood, mi_ood = uncertainty_scores(model, ood_loader, device, args.T)

    # OOD is the positive class (label 1)
    y_true = np.concatenate([np.zeros_like(ent_in), np.ones_like(ent_ood)])
    auroc_entropy = roc_auc_score(y_true, np.concatenate([ent_in, ent_ood]))
    auroc_mi = roc_auc_score(y_true, np.concatenate([mi_in, mi_ood]))

    print(f"Mean entropy  | MNIST (in): {ent_in.mean():.4f}   FashionMNIST (OOD): {ent_ood.mean():.4f}")
    print(f"AUROC (entropy as OOD score): {auroc_entropy:.4f}")
    print(f"AUROC (mutual information):   {auroc_mi:.4f}")

    summary = pd.DataFrame({
        "metric": [
            "Mean predictive entropy - MNIST (in-dist)",
            "Mean predictive entropy - FashionMNIST (OOD)",
            "Mean mutual information - MNIST (in-dist)",
            "Mean mutual information - FashionMNIST (OOD)",
            "OOD-detection AUROC (entropy)",
            "OOD-detection AUROC (mutual information)",
        ],
        "value": [
            f"{ent_in.mean():.4f}", f"{ent_ood.mean():.4f}",
            f"{mi_in.mean():.4f}", f"{mi_ood.mean():.4f}",
            f"{auroc_entropy:.4f}", f"{auroc_mi:.4f}",
        ],
    })
    summary.to_csv(f"{common.RES_DIR}/ood_summary.csv", index=False)
    print("Saved", f"{common.RES_DIR}/ood_summary.csv")
    print(summary.to_string(index=False))

    plt.figure(figsize=(7, 4))
    plt.hist(ent_in, bins=60, alpha=0.6, label="MNIST (in-dist)", density=True)
    plt.hist(ent_ood, bins=60, alpha=0.6, label="FashionMNIST (OOD)", density=True)
    plt.xlabel("Predictive entropy"); plt.ylabel("density")
    plt.title(f"MC Dropout entropy: in-dist vs OOD (AUROC={auroc_entropy:.3f})")
    plt.legend(); plt.tight_layout()
    out = f"{common.FIG_DIR}/ood_entropy_hist.png"
    plt.savefig(out, dpi=150)
    print("Saved", out)


if __name__ == "__main__":
    main()
