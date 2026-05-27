"""Experiment 2 — standard vs MC Dropout, and the uncertainty of correct vs
wrong predictions.

Produces:
  results/summary.csv
  figures/entropy_correct_vs_wrong.png
  figures/variance_correct_vs_wrong.png
  figures/confident_examples.png
  figures/uncertain_examples.png
"""

import argparse

import common  # noqa: F401
import matplotlib.pyplot as plt
import pandas as pd
import torch

from mc_dropout import (
    set_seed, get_device, MCDropoutCNN, get_mnist_loaders, get_raw_test_dataset,
    standard_predict, mc_forward, summarise,
)


def load_model(device):
    ckpt = torch.load(common.MODEL_PATH, map_location=device)
    p = ckpt.get("meta", {}).get("dropout", 0.5)
    model = MCDropoutCNN(p=p).to(device)
    model.load_state_dict(ckpt["state_dict"])
    return model


def show_examples(indices, raw_set, preds, entropy, title, fname):
    n = len(indices)
    fig, axes = plt.subplots(1, n, figsize=(2 * n, 2.6))
    if n == 1:
        axes = [axes]
    for ax, idx in zip(axes, indices):
        img, true = raw_set[idx]
        ax.imshow(img.squeeze().numpy(), cmap="gray")
        ax.set_title(f"pred={int(preds[idx])} (t={int(true)})\nH={entropy[idx]:.2f}", fontsize=9)
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(fname, dpi=150)
    print("Saved", fname)


def main():
    parser = argparse.ArgumentParser(description="Standard vs MC Dropout uncertainty")
    parser.add_argument("--T", type=int, default=30, help="number of MC forward passes")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device(args.device)
    print("Using device:", device)

    _, test_loader = get_mnist_loaders(batch_size=128, data_dir=common.DATA_DIR)
    raw_set = get_raw_test_dataset(data_dir=common.DATA_DIR)
    model = load_model(device)

    # Standard (dropout off)
    std_probs, labels = standard_predict(model, test_loader, device)
    std_acc = (std_probs.argmax(1) == labels).float().mean().item()
    print(f"Standard test accuracy: {std_acc:.4f}")

    # MC Dropout (dropout on, T passes)
    probs_T, labels = mc_forward(model, test_loader, device, T=args.T)
    out = summarise(probs_T, labels)
    preds, entropy, variance = out["preds"], out["entropy"], out["variance"]
    mc_acc = (preds == labels).float().mean().item()
    print(f"MC Dropout test accuracy (T={args.T}): {mc_acc:.4f}")

    correct_mask = preds == labels
    wrong_mask = ~correct_mask
    ent_c = entropy[correct_mask].mean().item()
    ent_w = entropy[wrong_mask].mean().item()
    var_c = variance[correct_mask].mean().item()
    var_w = variance[wrong_mask].mean().item()
    n_wrong = int(wrong_mask.sum())

    print(f"Mean entropy  | correct: {ent_c:.4f}   wrong: {ent_w:.4f}")
    print(f"Mean variance | correct: {var_c:.6f}  wrong: {var_w:.6f}")
    print(f"# wrong MC predictions: {n_wrong} / {len(labels)}")

    summary = pd.DataFrame({
        "metric": [
            "Standard test accuracy",
            f"MC Dropout test accuracy (T={args.T})",
            "Mean predictive entropy - correct",
            "Mean predictive entropy - wrong",
            "Mean predictive variance - correct",
            "Mean predictive variance - wrong",
            "Entropy ratio (wrong / correct)",
            "# wrong MC predictions",
        ],
        "value": [
            f"{std_acc:.4f}", f"{mc_acc:.4f}",
            f"{ent_c:.4f}", f"{ent_w:.4f}",
            f"{var_c:.6f}", f"{var_w:.6f}",
            f"{(ent_w / ent_c):.2f}",
            f"{n_wrong} / {len(labels)}",
        ],
    })
    summary.to_csv(f"{common.RES_DIR}/summary.csv", index=False)
    print("Saved", f"{common.RES_DIR}/summary.csv")
    print(summary.to_string(index=False))

    # Entropy histogram, correct vs wrong
    plt.figure(figsize=(7, 4))
    plt.hist(entropy[correct_mask].numpy(), bins=50, alpha=0.6, label="correct", density=True)
    plt.hist(entropy[wrong_mask].numpy(), bins=50, alpha=0.6, label="wrong", density=True)
    plt.xlabel("Predictive entropy"); plt.ylabel("density")
    plt.title("MC Dropout predictive entropy: correct vs wrong")
    plt.legend(); plt.tight_layout()
    plt.savefig(f"{common.FIG_DIR}/entropy_correct_vs_wrong.png", dpi=150)
    print("Saved", f"{common.FIG_DIR}/entropy_correct_vs_wrong.png")

    # Variance histogram, correct vs wrong
    plt.figure(figsize=(7, 4))
    plt.hist(variance[correct_mask].numpy(), bins=50, alpha=0.6, label="correct", density=True)
    plt.hist(variance[wrong_mask].numpy(), bins=50, alpha=0.6, label="wrong", density=True)
    plt.xlabel("Predictive variance"); plt.ylabel("density")
    plt.title("MC Dropout predictive variance: correct vs wrong")
    plt.legend(); plt.tight_layout()
    plt.savefig(f"{common.FIG_DIR}/variance_correct_vs_wrong.png", dpi=150)
    print("Saved", f"{common.FIG_DIR}/variance_correct_vs_wrong.png")

    # Most confident / most uncertain examples
    sorted_idx = torch.argsort(entropy)
    confident = sorted_idx[:8].tolist()
    uncertain = sorted_idx[-8:].tolist()
    show_examples(confident, raw_set, preds, entropy,
                  "Most CONFIDENT predictions (low entropy)",
                  f"{common.FIG_DIR}/confident_examples.png")
    show_examples(uncertain, raw_set, preds, entropy,
                  "Most UNCERTAIN predictions (high entropy)",
                  f"{common.FIG_DIR}/uncertain_examples.png")


if __name__ == "__main__":
    main()
