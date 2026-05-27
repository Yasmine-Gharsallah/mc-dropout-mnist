"""Experiment 3 — the paper's signature rotating-digit demonstration.

Take one test digit, rotate it through a range of angles, and run MC Dropout at
each angle. As the digit rotates into an ambiguous shape the predicted class
probabilities scatter and the predictive entropy rises.

Produces:
  figures/rotating_digit.png
  results/rotating_digit.csv
"""

import argparse

import common  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode

from mc_dropout import (
    set_seed, get_device, MCDropoutCNN, get_raw_test_dataset,
    enable_dropout, mc_forward_batch, normalise,
)

EPS = 1e-12


def load_model(device):
    ckpt = torch.load(common.MODEL_PATH, map_location=device)
    p = ckpt.get("meta", {}).get("dropout", 0.5)
    model = MCDropoutCNN(p=p).to(device)
    model.load_state_dict(ckpt["state_dict"])
    return model


def main():
    parser = argparse.ArgumentParser(description="Rotating-digit uncertainty")
    parser.add_argument("--digit", type=int, default=7, help="which digit class to rotate")
    parser.add_argument("--index", type=int, default=None, help="explicit test index (overrides --digit)")
    parser.add_argument("--T", type=int, default=50)
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
    disp_imgs = []
    mean_probs = np.zeros((len(angles), 10))
    std_probs = np.zeros((len(angles), 10))
    entropies = np.zeros(len(angles))

    with torch.no_grad():
        for i, ang in enumerate(angles):
            rotated_raw = TF.rotate(raw_img, ang, interpolation=InterpolationMode.BILINEAR, fill=0.0)
            disp_imgs.append(rotated_raw.squeeze().numpy())
            x = normalise(rotated_raw).unsqueeze(0).to(device)   # (1,1,28,28)
            probs_T = mc_forward_batch(model, x, T=args.T)        # (T,1,10)
            mp = probs_T.mean(0).squeeze(0).cpu().numpy()         # (10,)
            sp = probs_T.std(0).squeeze(0).cpu().numpy()          # (10,)
            mean_probs[i] = mp
            std_probs[i] = sp
            entropies[i] = float(-(mp * np.log(mp + EPS)).sum())

    # which classes to draw: the true label plus any class that gets noticeable mass
    selected = sorted(set([int(true_label)] + list(np.where(mean_probs.max(axis=0) > 0.05)[0])))
    cmap = plt.get_cmap("tab10")

    n = len(angles)
    fig = plt.figure(figsize=(max(9, n * 0.85), 6.2))
    gs = fig.add_gridspec(2, n, height_ratios=[1, 3], hspace=0.35)

    for i, ang in enumerate(angles):
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(disp_imgs[i], cmap="gray")
        ax.set_title(f"{ang}°", fontsize=8)
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
    out = f"{common.FIG_DIR}/rotating_digit.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print("Saved", out)

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
