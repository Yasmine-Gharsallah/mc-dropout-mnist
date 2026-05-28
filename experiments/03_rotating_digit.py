"""Experiment 3 - the paper's signature rotating-digit demonstration.

Take one test digit, rotate it through a range of angles, and run MC Dropout at
each angle. As the digit rotates into an ambiguous shape the predicted class
probabilities scatter and the predictive entropy rises.

This script produces two figures:
  figures/rotating_digit.png         - mean +/- 1 std softmax curves + entropy
  figures/rotating_digit_scatter.png - per-pass scatter of softmax INPUTS (logits)
                                       and softmax OUTPUTS, like Fig. 4 of the
                                       paper (Gal & Ghahramani, 2016).

CSV:
  results/rotating_digit.csv

Default digit is 1 to mirror the paper's Fig. 4 exactly.
"""

import argparse

import common  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode

from mc_dropout import (
    set_seed, get_device, MCDropoutCNN, get_raw_test_dataset,
    enable_dropout, normalise,
)

EPS = 1e-12


def load_model(device):
    ckpt = torch.load(common.MODEL_PATH, map_location=device)
    p = ckpt.get("meta", {}).get("dropout", 0.5)
    model = MCDropoutCNN(p=p).to(device)
    model.load_state_dict(ckpt["state_dict"])
    return model


@torch.no_grad()
def mc_logits_and_probs(model, x, T):
    """Run T stochastic passes and return (logits_T, probs_T), each of shape (T, C)."""
    logits = torch.stack([model(x) for _ in range(T)], dim=0).squeeze(1)  # (T, C)
    probs = F.softmax(logits, dim=-1)
    return logits.cpu().numpy(), probs.cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description="Rotating-digit uncertainty")
    parser.add_argument("--digit", type=int, default=1,
                        help="which digit class to rotate (default 1 mirrors paper Fig. 4).")
    parser.add_argument("--index", type=int, default=None,
                        help="explicit test index (overrides --digit)")
    parser.add_argument("--T", type=int, default=100,
                        help="number of MC forward passes (paper uses 100 for Fig. 4).")
    parser.add_argument("--min-angle", type=int, default=-90)
    parser.add_argument("--max-angle", type=int, default=90)
    parser.add_argument("--step", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device(args.device)
    print("Using device:", device)

    raw_set = get_raw_test_dataset(data_dir=common.DATA_DIR)
    model = load_model(device)
    model.eval()
    enable_dropout(model)

    # pick the image
    if args.index is not None:
        idx = args.index
    else:
        idx = next(i for i in range(len(raw_set)) if int(raw_set[i][1]) == args.digit)
    raw_img, true_label = raw_set[idx]
    print(f"Using test index {idx} (true label {int(true_label)})")

    angles = list(range(args.min_angle, args.max_angle + 1, args.step))
    n_ang = len(angles)
    C = 10

    disp_imgs = []
    all_logits = np.zeros((n_ang, args.T, C))   # softmax INPUTS  (per pass)
    all_probs = np.zeros((n_ang, args.T, C))    # softmax OUTPUTS (per pass)

    with torch.no_grad():
        for i, ang in enumerate(angles):
            rotated_raw = TF.rotate(raw_img, ang, interpolation=InterpolationMode.BILINEAR, fill=0.0)
            disp_imgs.append(rotated_raw.squeeze().numpy())
            x = normalise(rotated_raw).unsqueeze(0).to(device)  # (1, 1, 28, 28)
            logits_T, probs_T = mc_logits_and_probs(model, x, T=args.T)
            all_logits[i] = logits_T
            all_probs[i] = probs_T

    mean_probs = all_probs.mean(axis=1)                          # (n_ang, C)
    std_probs = all_probs.std(axis=1)                            # (n_ang, C)
    entropies = -(mean_probs * np.log(mean_probs + EPS)).sum(axis=1)

    # which classes to draw: true label plus any class with noticeable mass
    selected = sorted(set([int(true_label)] +
                          list(np.where(mean_probs.max(axis=0) > 0.05)[0])))
    cmap = plt.get_cmap("tab10")

    # ------------------------------------------------------------------ Figure 1
    # Mean +/- 1 std curves with entropy overlay (the original view).
    fig = plt.figure(figsize=(max(9, n_ang * 0.85), 6.2))
    gs = fig.add_gridspec(2, n_ang, height_ratios=[1, 3], hspace=0.35)
    for i, ang in enumerate(angles):
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(disp_imgs[i], cmap="gray")
        ax.set_title(f"{ang} deg", fontsize=8)
        ax.axis("off")
    ax = fig.add_subplot(gs[1, :])
    for c in selected:
        ax.plot(angles, mean_probs[:, c], marker="o", color=cmap(c % 10), label=f"class {c}")
        ax.fill_between(angles, mean_probs[:, c] - std_probs[:, c],
                        mean_probs[:, c] + std_probs[:, c], color=cmap(c % 10), alpha=0.15)
    ax.set_xlabel("rotation angle (degrees)")
    ax.set_ylabel("mean softmax probability")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    ax2 = ax.twinx()
    ax2.plot(angles, entropies, "k--", linewidth=2, label="entropy")
    ax2.set_ylabel("predictive entropy (nats)")
    ax2.legend(loc="upper right", fontsize=8)
    fig.suptitle(f"Rotating digit (true label {int(true_label)}): MC Dropout uncertainty")
    out1 = f"{common.FIG_DIR}/rotating_digit.png"
    fig.savefig(out1, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved", out1)

    # ------------------------------------------------------------------ Figure 2
    # Per-pass scatter, mirroring paper Fig. 4.
    fig2 = plt.figure(figsize=(max(11, n_ang * 0.95), 7.5))
    gs2 = fig2.add_gridspec(3, n_ang, height_ratios=[1, 4, 4], hspace=0.35)
    # row 0: thumbnails
    for i, ang in enumerate(angles):
        ax = fig2.add_subplot(gs2[0, i])
        ax.imshow(disp_imgs[i], cmap="gray")
        ax.set_title(f"{ang} deg", fontsize=8)
        ax.axis("off")

    # row 1: softmax INPUT (logits) scatter
    ax_in = fig2.add_subplot(gs2[1, :])
    # row 2: softmax OUTPUT scatter
    ax_out = fig2.add_subplot(gs2[2, :], sharex=ax_in)

    # jitter angles slightly per class so identical-x scatter points don't overlap
    jitter_step = (angles[1] - angles[0]) * 0.045 if n_ang > 1 else 0.5
    for c in selected:
        color = cmap(c % 10)
        for i, ang in enumerate(angles):
            jx = ang + (c - 4.5) * jitter_step
            xs = np.full(args.T, jx)
            ax_in.scatter(xs, all_logits[i, :, c], s=6, alpha=0.35,
                          color=color, edgecolors="none")
            ax_out.scatter(xs, all_probs[i, :, c], s=6, alpha=0.35,
                           color=color, edgecolors="none")
        # legend proxy
        ax_in.plot([], [], "o", color=color, label=f"class {c}")
    ax_in.set_ylabel("softmax input (logit)")
    ax_in.set_title("Per-pass softmax input scatter (paper Fig. 4a analogue)")
    ax_in.legend(loc="upper left", fontsize=8, ncol=min(len(selected), 5))
    ax_in.grid(True, linestyle=":", alpha=0.4)

    ax_out.set_ylabel("softmax output (probability)")
    ax_out.set_xlabel("rotation angle (degrees)")
    ax_out.set_title("Per-pass softmax output scatter (paper Fig. 4b analogue)")
    ax_out.set_ylim(-0.03, 1.03)
    ax_out.grid(True, linestyle=":", alpha=0.4)

    fig2.suptitle(f"Rotating digit (true label {int(true_label)}), T={args.T} stochastic passes")
    out2 = f"{common.FIG_DIR}/rotating_digit_scatter.png"
    fig2.savefig(out2, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print("Saved", out2)

    # ------------------------------------------------------------------ CSV
    df = pd.DataFrame({
        "angle": angles,
        "entropy": entropies,
        "top_class": mean_probs.argmax(axis=1),
        "top_prob": mean_probs.max(axis=1),
    })
    df.to_csv(f"{common.RES_DIR}/rotating_digit.csv", index=False)
    print("Saved", f"{common.RES_DIR}/rotating_digit.csv")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
