"""Experiment 7 - reproduce one UCI-regression row from Table 1 of Gal &
Ghahramani (2016), Section 5.3.

By default we use Concrete Compressive Strength (1030 samples, 8 features).
The paper reports for this dataset (Table 1):
    RMSE 5.23 +/- 0.12   test LL -3.04 +/- 0.02   (Dropout, 1 layer, 40 epochs)
And in the updated Table 2 (10x epochs, still 1 layer):
    RMSE 4.81 +/- 0.14   test LL -2.94 +/- 0.02

We use the same architecture (1 hidden layer, 50 units), the same dropout
philosophy (small p for small datasets), Adam + batch 32, and choose the
model precision tau by a validation grid search (a cheaper stand-in for the
paper's Bayesian optimisation; both pick tau to maximise validation LL).

Produces:
  results/uci_regression.csv   - per-split + mean +/- std metrics
  figures/uci_pred_vs_true.png - sanity scatter of the last split
"""

import argparse

import common  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from mc_dropout import set_seed, get_device
from mc_dropout.uci import DropoutMLP, train_mlp, mc_predict, mc_metrics, search_tau


def load_concrete():
    """Concrete Compressive Strength via OpenML (mirrors the paper's row).

    OpenML's "Concrete_Data" dataset (id 4353) ships without a default target
    attribute, so we treat the last column ("Concrete compressive strength")
    as the regression target ourselves.
    """
    from sklearn.datasets import fetch_openml
    ds = fetch_openml(name="Concrete_Data", version=1, as_frame=True,
                      data_home=common.DATA_DIR)
    df = ds.frame
    X = df.iloc[:, :-1].to_numpy(dtype=np.float64)
    y = df.iloc[:, -1].to_numpy(dtype=np.float64)
    return X, y, "Concrete Strength"


def load_energy():
    """Energy Efficiency (Y1 = heating load) via OpenML."""
    from sklearn.datasets import fetch_openml
    ds = fetch_openml(name="ENB2012_data", version=1, as_frame=True,
                      data_home=common.DATA_DIR)
    df = ds.frame
    # Pick Y1 (heating load) as the regression target, drop both targets from X
    # if present; otherwise fall back to "the last column is the target".
    target_cols = [c for c in df.columns if c.upper().startswith("Y")]
    if "Y1" in df.columns:
        y = df["Y1"].to_numpy(dtype=np.float64)
    else:
        y = df.iloc[:, -1].to_numpy(dtype=np.float64)
    X = df.drop(columns=target_cols).to_numpy(dtype=np.float64) if target_cols \
        else df.iloc[:, :-1].to_numpy(dtype=np.float64)
    return X, y, "Energy Efficiency (Y1)"


DATASETS = {"concrete": load_concrete, "energy": load_energy}


def run_split(X_tr, y_tr, X_te, y_te, p, epochs, T, device):
    """Single split: train, MC-predict, search tau on test set (paper does on val,
    we use a held-out 20% of train as a tiny val set for tau selection)."""
    # Standardise using TRAIN statistics only.
    x_mean, x_std = X_tr.mean(0), X_tr.std(0) + 1e-8
    y_mean, y_std = float(y_tr.mean()), float(y_tr.std() + 1e-8)
    X_tr_n = (X_tr - x_mean) / x_std
    X_te_n = (X_te - x_mean) / x_std
    y_tr_n = (y_tr - y_mean) / y_std
    y_te_n = (y_te - y_mean) / y_std

    # Carve a small validation slice (20%) out of train for tau selection.
    n_val = max(1, int(0.2 * len(X_tr_n)))
    X_val_n, y_val_n = X_tr_n[-n_val:], y_tr_n[-n_val:]
    X_fit_n, y_fit_n = X_tr_n[:-n_val], y_tr_n[:-n_val]

    model = DropoutMLP(in_dim=X_tr_n.shape[1], hidden=50, p=p).to(device)
    train_mlp(model, X_fit_n, y_fit_n, epochs=epochs, device=device)

    val_preds = mc_predict(model, X_val_n, T=T, device=device)
    tau = search_tau(val_preds, y_val_n, y_mean, y_std)

    test_preds = mc_predict(model, X_te_n, T=T, device=device)
    rmse, ll = mc_metrics(test_preds, y_te_n, y_mean, y_std, tau)
    return rmse, ll, tau, test_preds, y_te_n, y_mean, y_std


def main():
    parser = argparse.ArgumentParser(description="UCI regression via MC Dropout (Table 1)")
    parser.add_argument("--dataset", choices=list(DATASETS), default="concrete")
    parser.add_argument("--n-splits", type=int, default=5,
                        help="Random 90/10 splits (paper: 20; reduced here for compute).")
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--T", type=int, default=1000,
                        help="MC forward passes per evaluation.")
    parser.add_argument("--p", type=float, default=0.05,
                        help="Dropout probability (paper: 0.05 for small datasets).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device(args.device)
    print(f"Using device: {device}  dataset: {args.dataset}  splits: {args.n_splits}")

    X, y, name = DATASETS[args.dataset]()
    print(f"Loaded {name}: X={X.shape}  y={y.shape}  "
          f"y range=[{y.min():.2f}, {y.max():.2f}]")

    rows = []
    last_split_artifacts = None
    for s in range(args.n_splits):
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.1, random_state=args.seed + s,
        )
        rmse, ll, tau, preds_T, y_te_n, y_mean, y_std = run_split(
            X_tr, y_tr, X_te, y_te,
            p=args.p, epochs=args.epochs, T=args.T, device=device,
        )
        print(f"  split {s}: tau={tau:<6}  RMSE={rmse:.3f}  LL={ll:.3f}")
        rows.append({"split": s, "tau": tau, "rmse": rmse, "log_likelihood": ll})
        last_split_artifacts = (preds_T, y_te_n, y_mean, y_std)

    df = pd.DataFrame(rows)
    summary = pd.DataFrame([
        {"split": "mean", "tau": float(df["tau"].mean()),
         "rmse": float(df["rmse"].mean()), "log_likelihood": float(df["log_likelihood"].mean())},
        {"split": "std",  "tau": float(df["tau"].std(ddof=0)),
         "rmse": float(df["rmse"].std(ddof=0)),
         "log_likelihood": float(df["log_likelihood"].std(ddof=0))},
    ])
    df_out = pd.concat([df, summary], ignore_index=True)
    # round for readability
    for c in ("tau", "rmse", "log_likelihood"):
        df_out[c] = df_out[c].astype(float).round(4)
    out_csv = f"{common.RES_DIR}/uci_regression.csv"
    df_out.to_csv(out_csv, index=False)
    print("Saved", out_csv)
    print(df_out.to_string(index=False))

    # Sanity scatter: predicted (mean over T) vs true on the last split.
    preds_T, y_te_n, y_mean, y_std = last_split_artifacts
    preds = preds_T * y_std + y_mean
    y_true = y_te_n * y_std + y_mean
    mean_pred = preds.mean(axis=0)
    std_pred = preds.std(axis=0)
    plt.figure(figsize=(6, 5))
    plt.errorbar(y_true, mean_pred, yerr=2 * std_pred, fmt="o", alpha=0.7,
                 markersize=4, ecolor="lightgray", elinewidth=1, capsize=2,
                 label="MC mean +/- 2 std")
    lims = [min(y_true.min(), mean_pred.min()) - 1, max(y_true.max(), mean_pred.max()) + 1]
    plt.plot(lims, lims, "k--", linewidth=1, label="y = x")
    plt.xlim(lims); plt.ylim(lims)
    plt.xlabel("true target"); plt.ylabel("predicted target")
    plt.title(f"{name}: MC Dropout predictions on the last split")
    plt.legend()
    plt.tight_layout()
    out_fig = f"{common.FIG_DIR}/uci_pred_vs_true.png"
    plt.savefig(out_fig, dpi=150)
    print("Saved", out_fig)


if __name__ == "__main__":
    main()
