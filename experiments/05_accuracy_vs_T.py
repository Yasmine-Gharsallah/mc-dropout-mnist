"""Experiment 5 — how the MC Dropout estimate converges with the number of
forward passes T.

We run T_max stochastic passes once, then evaluate the running estimate using the
first t passes for several values of t. As t grows, accuracy and negative
log-likelihood stabilise.

Produces:
  figures/accuracy_vs_T.png
  results/accuracy_vs_T.csv
"""

import argparse

import common  # noqa: F401
import matplotlib.pyplot as plt
import pandas as pd
import torch

from mc_dropout import (
    set_seed, get_device, MCDropoutCNN, get_mnist_loaders,
    standard_predict, mc_forward, negative_log_likelihood,
)


def load_model(device):
    ckpt = torch.load(common.MODEL_PATH, map_location=device)
    p = ckpt.get("meta", {}).get("dropout", 0.5)
    model = MCDropoutCNN(p=p).to(device)
    model.load_state_dict(ckpt["state_dict"])
    return model


def main():
    parser = argparse.ArgumentParser(description="Accuracy / NLL vs number of MC samples")
    parser.add_argument("--max-T", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device(args.device)
    print("Using device:", device)

    _, test_loader = get_mnist_loaders(batch_size=128, data_dir=common.DATA_DIR)
    model = load_model(device)

    std_probs, labels = standard_predict(model, test_loader, device)
    std_acc = (std_probs.argmax(1) == labels).float().mean().item()

    probs_T, labels = mc_forward(model, test_loader, device, T=args.max_T)

    candidate_ts = [1, 2, 3, 5, 10, 20, 30, 50, 75, 100]
    t_values = [t for t in candidate_ts if t <= args.max_T]

    rows = []
    for t in t_values:
        mean_probs = probs_T[:t].mean(dim=0)
        preds = mean_probs.argmax(dim=1)
        acc = (preds == labels).float().mean().item()
        nll = negative_log_likelihood(mean_probs, labels).mean().item()
        rows.append({"T": t, "accuracy": acc, "nll": nll})
        print(f"T={t:3d}  acc={acc:.4f}  nll={nll:.4f}")

    df = pd.DataFrame(rows)
    df.to_csv(f"{common.RES_DIR}/accuracy_vs_T.csv", index=False)
    print("Saved", f"{common.RES_DIR}/accuracy_vs_T.csv")

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(df["T"], df["accuracy"], marker="o", label="MC Dropout")
    ax[0].axhline(std_acc, color="gray", linestyle="--", label="standard (dropout off)")
    ax[0].set_xlabel("number of MC samples T"); ax[0].set_ylabel("test accuracy")
    ax[0].set_title("Accuracy vs T"); ax[0].legend()

    ax[1].plot(df["T"], df["nll"], marker="o", color="tab:red")
    ax[1].set_xlabel("number of MC samples T"); ax[1].set_ylabel("mean NLL")
    ax[1].set_title("Negative log-likelihood vs T")

    fig.tight_layout()
    out = f"{common.FIG_DIR}/accuracy_vs_T.png"
    fig.savefig(out, dpi=150)
    print("Saved", out)


if __name__ == "__main__":
    main()
