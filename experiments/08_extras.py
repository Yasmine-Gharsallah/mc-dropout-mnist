"""Experiment 8 - extra analyses produced from the trained checkpoint.

This script does not retrain anything. It loads results/model.pt, runs a
single MC pass over MNIST and FashionMNIST, and produces five additional
figures and three CSVs that enrich the report:

  figures/reliability_diagram.png       - calibration curve (ECE table)
  figures/rejection_curve.png           - accuracy vs coverage when sorting by entropy
  figures/uncertainty_decomposition.png - H[E[p]] vs E[H[p]] (total / aleatoric / epistemic)
  figures/wrong_confusion_matrix.png    - confusion matrix restricted to MC-wrong predictions
  figures/ood_score_comparison.png      - ROC curves for 4 OOD scores (entropy, MI, max-softmax, variance)

  results/calibration.csv               - ECE per method (std vs MC), bin-wise stats
  results/rejection.csv                 - accuracy at each retained-fraction
  results/ood_score_comparison.csv      - AUROC per OOD score
"""

import common  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve

from mc_dropout import (
    set_seed, get_device, MCDropoutCNN,
    get_mnist_loaders, get_ood_loader,
    standard_predict, mc_forward,
    predictive_mean, predictive_entropy, expected_entropy,
    mutual_information, predictive_variance,
)


def load_model(device):
    ckpt = torch.load(common.MODEL_PATH, map_location=device)
    p = ckpt.get("meta", {}).get("dropout", 0.5)
    model = MCDropoutCNN(p=p).to(device)
    model.load_state_dict(ckpt["state_dict"])
    return model


def expected_calibration_error(conf, correct, n_bins=15):
    """ECE: weighted mean |accuracy_in_bin - confidence_in_bin| over equal-width
    confidence bins on [0, 1]. Returns (ECE, dataframe of bin stats)."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    ece = 0.0
    n = len(conf)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        cnt = int(mask.sum())
        if cnt == 0:
            rows.append({"bin_lo": lo, "bin_hi": hi, "count": 0,
                         "avg_conf": np.nan, "accuracy": np.nan, "gap": np.nan})
            continue
        avg_c = float(conf[mask].mean())
        acc = float(correct[mask].mean())
        gap = abs(avg_c - acc)
        ece += (cnt / n) * gap
        rows.append({"bin_lo": lo, "bin_hi": hi, "count": cnt,
                     "avg_conf": avg_c, "accuracy": acc, "gap": gap})
    return ece, pd.DataFrame(rows)


def main():
    set_seed(42)
    device = get_device("auto")
    print("Using device:", device)

    _, test_loader = get_mnist_loaders(batch_size=128, data_dir=common.DATA_DIR)
    ood_loader = get_ood_loader(batch_size=128, data_dir=common.DATA_DIR)
    model = load_model(device)

    # ----------------------------------------------------------------- inference
    std_probs, labels = standard_predict(model, test_loader, device)
    probs_T, labels = mc_forward(model, test_loader, device, T=30)
    mean_probs = predictive_mean(probs_T)

    std_preds = std_probs.argmax(1)
    mc_preds = mean_probs.argmax(1)
    std_correct = (std_preds == labels).numpy()
    mc_correct = (mc_preds == labels).numpy()

    std_conf = std_probs.max(1).values.numpy()  # max-softmax
    mc_conf = mean_probs.max(1).values.numpy()

    entropy = predictive_entropy(mean_probs).numpy()
    mi = mutual_information(probs_T).numpy()
    aleatoric = expected_entropy(probs_T).numpy()
    variance = predictive_variance(probs_T).numpy()

    # ============================================================ (1) calibration
    ece_std, bins_std = expected_calibration_error(std_conf, std_correct)
    ece_mc, bins_mc = expected_calibration_error(mc_conf, mc_correct)
    print(f"ECE  | standard: {ece_std:.4f}   MC Dropout: {ece_mc:.4f}")

    cal_df = pd.concat([bins_std.assign(method="standard"),
                        bins_mc.assign(method="mc_dropout")], ignore_index=True)
    cal_df.to_csv(f"{common.RES_DIR}/calibration.csv", index=False)

    # Reliability diagram
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, (m, bins, ece, label) in zip(
        axes,
        [("standard", bins_std, ece_std, f"Standard (ECE={ece_std:.3f})"),
         ("mc_dropout", bins_mc, ece_mc, f"MC Dropout (ECE={ece_mc:.3f})")],
    ):
        x = (bins["bin_lo"] + bins["bin_hi"]) / 2.0
        gap_bar = bins["avg_conf"] - bins["accuracy"]
        ax.bar(x, bins["accuracy"], width=1.0 / len(bins) * 0.9,
               color="tab:blue", alpha=0.7, edgecolor="black", label="accuracy")
        # gap bars in red, drawn on top
        ax.bar(x, gap_bar, width=1.0 / len(bins) * 0.9,
               bottom=bins["accuracy"], color="tab:red", alpha=0.55,
               edgecolor="black", label="gap (confidence - accuracy)")
        ax.plot([0, 1], [0, 1], "k--", linewidth=1)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xlabel("predicted confidence"); ax.set_title(label)
        ax.legend(loc="upper left", fontsize=8)
    axes[0].set_ylabel("empirical accuracy")
    fig.suptitle("Reliability diagrams: 15 equal-width confidence bins")
    fig.tight_layout()
    out = f"{common.FIG_DIR}/reliability_diagram.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("Saved", out)

    # ============================================================ (2) rejection / selective prediction
    # Sort test set by ascending entropy; for each coverage c in (0, 1], compute
    # accuracy on the most-confident c-fraction.
    order = np.argsort(entropy)
    n = len(labels)
    cov_grid = np.linspace(0.05, 1.0, 20)
    sel_rows = []
    for c in cov_grid:
        k = max(1, int(round(c * n)))
        idx = order[:k]
        acc = float(np.mean(mc_correct[idx]))
        sel_rows.append({"coverage": float(c), "accuracy": acc, "kept": k})
    sel_df = pd.DataFrame(sel_rows)
    sel_df.to_csv(f"{common.RES_DIR}/rejection.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(sel_df["coverage"], sel_df["accuracy"], marker="o", color="tab:blue",
            label="MC Dropout (sorted by entropy)")
    ax.axhline(mc_correct.mean(), color="gray", linestyle="--", linewidth=1,
               label=f"full-coverage accuracy = {mc_correct.mean():.4f}")
    # Same curve sorted by standard max-softmax for comparison
    order_std = np.argsort(-std_conf)
    accs_std = [float(np.mean(std_correct[order_std[:max(1, int(round(c * n)))]])) for c in cov_grid]
    ax.plot(cov_grid, accs_std, marker="s", color="tab:red", alpha=0.7,
            label="Standard (sorted by max-softmax)")
    ax.set_xlabel("coverage (fraction of test set retained)")
    ax.set_ylabel("accuracy on retained subset")
    ax.set_title("Selective prediction: accept the most-confident inputs first")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="lower left", fontsize=9)
    fig.tight_layout()
    out = f"{common.FIG_DIR}/rejection_curve.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("Saved", out)

    # ============================================================ (3) uncertainty decomposition
    correct_mask = mc_correct.astype(bool)
    wrong_mask = ~correct_mask
    decomp = pd.DataFrame({
        "predictions": ["correct", "wrong"],
        "total H[E[p]]": [float(entropy[correct_mask].mean()), float(entropy[wrong_mask].mean())],
        "aleatoric E[H[p]]": [float(aleatoric[correct_mask].mean()), float(aleatoric[wrong_mask].mean())],
        "epistemic (MI)": [float(mi[correct_mask].mean()), float(mi[wrong_mask].mean())],
    })
    print(decomp.to_string(index=False))

    fig, ax = plt.subplots(figsize=(7.5, 4))
    x = np.arange(2)
    width = 0.25
    ax.bar(x - width, decomp["total H[E[p]]"], width, label="total H[E[p]]")
    ax.bar(x, decomp["aleatoric E[H[p]]"], width, label="aleatoric E[H[p]]")
    ax.bar(x + width, decomp["epistemic (MI)"], width, label="epistemic (MI)")
    ax.set_xticks(x); ax.set_xticklabels(decomp["predictions"])
    ax.set_ylabel("mean uncertainty (nats)")
    ax.set_title("Uncertainty decomposition: aleatoric + epistemic")
    ax.legend()
    ax.grid(True, axis="y", linestyle=":", alpha=0.5)
    fig.tight_layout()
    out = f"{common.FIG_DIR}/uncertainty_decomposition.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("Saved", out)

    # ============================================================ (4) confusion of wrong predictions
    cm = confusion_matrix(labels.numpy()[wrong_mask], mc_preds.numpy()[wrong_mask],
                          labels=list(range(10)))
    fig, ax = plt.subplots(figsize=(6.2, 5.5))
    im = ax.imshow(cm, cmap="Reds")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="count")
    ax.set_xticks(range(10)); ax.set_yticks(range(10))
    ax.set_xlabel("predicted digit"); ax.set_ylabel("true digit")
    ax.set_title(f"Confusion of {int(wrong_mask.sum())} MC-wrong predictions")
    for i in range(10):
        for j in range(10):
            if cm[i, j] > 0:
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() * 0.5 else "black", fontsize=8)
    fig.tight_layout()
    out = f"{common.FIG_DIR}/wrong_confusion_matrix.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("Saved", out)

    # ============================================================ (5) OOD score comparison
    # Re-run a single MC pass over both loaders to score each input under 4 measures.
    probs_T_ood, _ = mc_forward(model, ood_loader, device, T=30)
    mean_probs_ood = predictive_mean(probs_T_ood)

    scores = {
        "predictive entropy": (entropy, predictive_entropy(mean_probs_ood).numpy()),
        "mutual information (BALD)": (mi, mutual_information(probs_T_ood).numpy()),
        "1 - max softmax (MC)": (1.0 - mean_probs.max(1).values.numpy(),
                                 1.0 - mean_probs_ood.max(1).values.numpy()),
        "predictive variance": (variance, predictive_variance(probs_T_ood).numpy()),
    }
    y_true = np.concatenate([np.zeros(len(labels)), np.ones(probs_T_ood.shape[1])])

    rows = []
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, (s_in, s_ood) in scores.items():
        scores_all = np.concatenate([s_in, s_ood])
        auroc = roc_auc_score(y_true, scores_all)
        fpr, tpr, _ = roc_curve(y_true, scores_all)
        ax.plot(fpr, tpr, label=f"{name} (AUROC={auroc:.3f})")
        rows.append({"score": name, "AUROC": auroc})
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.6, label="chance")
    ax.set_xlabel("false positive rate (in-dist accepted as OOD)")
    ax.set_ylabel("true positive rate (OOD detected)")
    ax.set_title("OOD detection: MNIST vs FashionMNIST")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, linestyle=":", alpha=0.5)
    fig.tight_layout()
    out = f"{common.FIG_DIR}/ood_score_comparison.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("Saved", out)

    ood_score_df = pd.DataFrame(rows)
    ood_score_df.to_csv(f"{common.RES_DIR}/ood_score_comparison.csv", index=False)
    print(ood_score_df.to_string(index=False))


if __name__ == "__main__":
    main()
