"""Experiment 6 — repeat the headline experiment over several seeds.

Trains a fresh model per seed and reports mean +/- std of the key metrics, so the
results come with error bars instead of resting on a single lucky run.

Produces:
  results/multiseed_stats.csv
"""

import argparse

import common  # noqa: F401
import numpy as np
import pandas as pd

from mc_dropout import (
    set_seed, get_device, MCDropoutCNN, get_mnist_loaders,
    train_model, standard_predict, mc_forward, summarise,
)


def run_one_seed(seed, epochs, dropout, T, device):
    set_seed(seed)
    train_loader, test_loader = get_mnist_loaders(batch_size=128, data_dir=common.DATA_DIR)
    model = MCDropoutCNN(p=dropout).to(device)
    train_model(model, train_loader, epochs=epochs, lr=1e-3, device=device, verbose=False)

    std_probs, labels = standard_predict(model, test_loader, device)
    std_acc = (std_probs.argmax(1) == labels).float().mean().item()

    probs_T, labels = mc_forward(model, test_loader, device, T=T)
    out = summarise(probs_T, labels)
    preds, entropy = out["preds"], out["entropy"]
    mc_acc = (preds == labels).float().mean().item()
    correct = preds == labels
    ent_c = entropy[correct].mean().item()
    ent_w = entropy[~correct].mean().item()
    return {
        "seed": seed,
        "std_acc": std_acc,
        "mc_acc": mc_acc,
        "entropy_correct": ent_c,
        "entropy_wrong": ent_w,
        "entropy_ratio": ent_w / ent_c,
    }


def main():
    parser = argparse.ArgumentParser(description="Multi-seed statistics")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--epochs", type=int, default=3, help="epochs per seed (kept small)")
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--T", type=int, default=30)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()

    device = get_device(args.device)
    print("Using device:", device, "| seeds:", args.seeds)

    rows = []
    for seed in args.seeds:
        print(f"\n=== seed {seed} ===")
        row = run_one_seed(seed, args.epochs, args.dropout, args.T, device)
        print(row)
        rows.append(row)

    df = pd.DataFrame(rows)
    metric_cols = ["std_acc", "mc_acc", "entropy_correct", "entropy_wrong", "entropy_ratio"]
    mean_row = {"seed": "mean", **{c: df[c].mean() for c in metric_cols}}
    std_row = {"seed": "std", **{c: df[c].std(ddof=0) for c in metric_cols}}
    df_out = pd.concat([df, pd.DataFrame([mean_row, std_row])], ignore_index=True)

    # round for readability
    for c in metric_cols:
        df_out[c] = df_out[c].astype(float).round(4)

    df_out.to_csv(f"{common.RES_DIR}/multiseed_stats.csv", index=False)
    print("\nSaved", f"{common.RES_DIR}/multiseed_stats.csv")
    print(df_out.to_string(index=False))


if __name__ == "__main__":
    main()
