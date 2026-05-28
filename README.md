# MC Dropout for Uncertainty Estimation on MNIST

A faithful reproduction of the practical core of:

> **Yarin Gal and Zoubin Ghahramani.** *Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning.* ICML 2016. [arXiv:1506.02142](https://arxiv.org/abs/1506.02142)

prepared for the **Advanced Statistical Inference 2026** course assignment.

The paper proves that a neural network trained with dropout is, mathematically, an approximation to a Bayesian model (variational inference in a deep Gaussian process). The practical consequence is **Monte Carlo (MC) Dropout**: keep dropout **active at test time**, run `T` stochastic forward passes per input, and use the spread of the predictions as a principled estimate of model uncertainty.

This repository reproduces the paper's signature MNIST classification result, reproduces one row of its UCI regression table (Concrete Compressive Strength), and adds quantitative analyses the original paper does not report (calibration, out-of-distribution detection, selective prediction, aleatoric/epistemic decomposition).

A typeset 9-page report (`report.pdf`, NeurIPS 2026 style) and a self-contained narrative notebook (`reproduction.ipynb`) are included.

---

## Table of contents

1. [What this project covers](#what-this-project-covers)
2. [Headline results](#headline-results)
3. [Repository layout](#repository-layout)
4. [Quick start](#quick-start)
5. [Running the experiments](#running-the-experiments)
6. [The notebook (`reproduction.ipynb`)](#the-notebook-reproductionipynb)
7. [The report (`report.pdf` / `report.tex`)](#the-report-reportpdf--reporttex)
8. [Implementation notes](#implementation-notes)
9. [Reproducibility](#reproducibility)

---

## What this project covers

| # | Experiment | Script | Paper claim it tests |
|---|---|---|---|
| 1 | Train a small CNN with dropout on MNIST. | `experiments/01_train.py` | architecture & training recipe |
| 2 | Standard vs MC Dropout test accuracy + entropy/variance/MI of correct vs wrong predictions. | `experiments/02_uncertainty.py` | MC Dropout preserves accuracy and produces meaningful uncertainty |
| 3 | **Rotating-digit demonstration** (paper Fig. 4): rotate the digit "1" through ±90°, run T=100 MC passes, plot both mean ± std curves and a per-pass scatter of softmax inputs and outputs. | `experiments/03_rotating_digit.py` | signature classification figure |
| 4 | Out-of-distribution (OOD) detection: MNIST vs FashionMNIST, scored with AUROC under predictive entropy. | `experiments/04_ood.py` | *extension* — paper's uncertainty also flags OOD inputs |
| 5 | Accuracy / NLL convergence as a function of the number of MC samples T (1 → 100). | `experiments/05_accuracy_vs_T.py` | "T=10 already gives a reasonable estimate" |
| 6 | Multi-seed robustness: retrain across several seeds, report mean ± std. | `experiments/06_multiseed.py` | rigor — the result is not seed-dependent |
| 7 | **UCI regression** (Concrete Compressive Strength) with the paper's setup (1-hidden-layer MLP, τ chosen by validation log-likelihood, eq. (8) for the test log-likelihood). | `experiments/07_uci_regression.py` | reproduces one row of paper Table 1 |
| 8 | **Extra analyses**: reliability diagram + ECE, selective prediction curve, aleatoric/epistemic decomposition, confusion of wrong predictions, comparison of four MC-derived OOD scores. | `experiments/08_extras.py` | *extensions* — calibration & comparative properties of the MC uncertainty signal |

All eight produce figures into `figures/` and CSV summaries into `results/`. Together they fully populate the report and the notebook.

---

## Headline results

| Metric | Value | Paper |
|---|---|---|
| MNIST standard test accuracy | **98.97 %** | qualitative |
| MNIST MC Dropout accuracy (T=30) | **98.99 %** | qualitative (accuracy preserved) ✓ |
| Mean predictive entropy — correct | 0.100 | low ✓ |
| Mean predictive entropy — wrong | 0.966 (≈ 9.7× higher) | high ✓ |
| Multi-seed MC accuracy (3 seeds) | 0.9884 ± 0.0007 | — |
| Multi-seed entropy ratio (wrong/correct) | 9.03 ± 0.10 | — |
| Rotating digit "1" — entropy @ 0° | 0.02 nats | matches Fig. 4 ✓ |
| Rotating digit "1" — entropy @ ±60° | 1.46 – 1.90 nats | matches Fig. 4 ✓ |
| OOD AUROC — predictive entropy | **0.988** | not in paper |
| OOD AUROC — mutual information (BALD) | 0.959 | not in paper |
| OOD AUROC — 1 − max softmax (MC) | 0.983 | not in paper |
| OOD AUROC — predictive variance | 0.948 | not in paper |
| Expected calibration error — standard | **0.002** | not in paper |
| Expected calibration error — MC Dropout | 0.019 (mildly under-confident) | not in paper |
| **UCI Concrete Strength — RMSE** | **5.63 ± 0.49** (5 splits) | 5.23 ± 0.12 (20 splits) |
| **UCI Concrete Strength — test log-likelihood** | **−3.14 ± 0.09** | −3.04 ± 0.02 |

See `report.pdf` for the full analysis and `results/*.csv` for the raw numbers.

---

## Repository layout

```
mc-dropout-mnist/
├── README.md
├── requirements.txt
├── reproduction.ipynb              # self-contained narrative notebook (runs end-to-end)
├── notebook.ipynb                  # thin runner over the CLI experiment scripts (legacy)
├── report.tex                      # NeurIPS 2026 source, single-author
├── report.pdf                      # 9-page typeset report (body 5p + appendix 4p)
├── neurips_2026.sty                # NeurIPS style file (adapted from official 2025)
│
├── src/mc_dropout/                 # reusable library code
│   ├── __init__.py
│   ├── utils.py                    # seeding, device selection
│   ├── model.py                    # MCDropoutCNN (classification, dropout p=0.5)
│   ├── data.py                     # MNIST loaders, rotation, OOD (FashionMNIST)
│   ├── mcdropout.py                # enable_dropout, standard/MC prediction, uncertainty metrics
│   ├── train.py                    # classification training loop, checkpointing
│   └── uci.py                      # UCI-regression MLP, MC predict, eq. (8) log-likelihood
│
├── experiments/                    # one script per claim; each saves figures + CSVs
│   ├── common.py                   # shared setup (paths, headless matplotlib)
│   ├── 01_train.py                 # train MNIST CNN
│   ├── 02_uncertainty.py           # standard vs MC, entropy/variance/MI of correct vs wrong
│   ├── 03_rotating_digit.py        # paper Fig. 4 mirror (mean-band + per-pass scatter)
│   ├── 04_ood.py                   # MNIST vs FashionMNIST OOD detection
│   ├── 05_accuracy_vs_T.py         # accuracy/NLL vs number of MC samples T
│   ├── 06_multiseed.py             # multi-seed mean ± std
│   ├── 07_uci_regression.py        # paper Table 1 (Concrete Compressive Strength)
│   ├── 08_extras.py                # calibration, rejection, decomposition, confusion, OOD scores
│   └── _build_notebook.py          # regenerates reproduction.ipynb from a Python source
│
├── figures/                        # generated plots (10 PNGs)
│   ├── training_curves.png
│   ├── entropy_correct_vs_wrong.png
│   ├── variance_correct_vs_wrong.png
│   ├── confident_examples.png
│   ├── uncertain_examples.png
│   ├── rotating_digit.png           # mean ± std curves + entropy overlay
│   ├── rotating_digit_scatter.png   # per-pass scatter (paper Fig. 4 mirror)
│   ├── ood_entropy_hist.png
│   ├── ood_score_comparison.png     # ROC curves for 4 MC-derived OOD scores
│   ├── accuracy_vs_T.png
│   ├── reliability_diagram.png      # calibration: standard vs MC, ECE table
│   ├── rejection_curve.png          # selective prediction (sort by entropy)
│   ├── uncertainty_decomposition.png  # total / aleatoric / epistemic
│   ├── wrong_confusion_matrix.png   # confusion of MC-wrong predictions
│   └── uci_pred_vs_true.png         # MC mean ± 2 std vs true target (Concrete)
│
└── results/                        # generated CSV summaries (7 files)
    ├── summary.csv                   # std vs MC accuracy + entropy/variance gaps
    ├── ood_summary.csv               # MNIST vs FashionMNIST entropy + AUROC
    ├── rotating_digit.csv            # per-angle entropy and top class
    ├── accuracy_vs_T.csv             # accuracy & NLL at each T ∈ {1,…,100}
    ├── multiseed_stats.csv           # per-seed metrics + mean/std rows
    ├── uci_regression.csv            # per-split RMSE & log-lik + mean/std
    ├── calibration.csv               # bin-wise ECE table
    ├── rejection.csv                 # accuracy at each retained-fraction
    └── ood_score_comparison.csv      # AUROC per OOD score
```

Folders **deliberately not in git**: `data/` (MNIST/FashionMNIST download cache), `*.pt` model checkpoints, `.venv/`, `.claude/`, LaTeX build artefacts (`*.aux`, `*.log`, …). All regenerated by the scripts.

---

## Quick start

```bash
# 1. Clone and enter
git clone https://github.com/Yasmine-Gharsallah/mc-dropout-mnist.git
cd mc-dropout-mnist

# 2. Create a virtual environment and install dependencies
python -m venv .venv
.venv/Scripts/activate            # Windows (PowerShell: .venv\Scripts\Activate.ps1)
# source .venv/bin/activate       # macOS / Linux
pip install -r requirements.txt

# 3. Either run the notebook end-to-end…
pip install jupyter
jupyter lab reproduction.ipynb

# 4. …or run the CLI pipeline (mirrors the notebook)
python experiments/01_train.py --epochs 5
python experiments/02_uncertainty.py --T 30
python experiments/03_rotating_digit.py --digit 1 --T 100
python experiments/04_ood.py --T 30
python experiments/05_accuracy_vs_T.py --max-T 100
python experiments/06_multiseed.py --seeds 0 1 2 --epochs 3
python experiments/07_uci_regression.py --dataset concrete --n-splits 5
python experiments/08_extras.py
```

A GPU is **not** required. End-to-end CPU runtime (Intel laptop class): training ≈ 1 min/epoch, all classification experiments ≈ 12 min, UCI regression ≈ 4 min, multi-seed (three retrains) ≈ 6 min.

---

## Running the experiments

Every script accepts `--help`. The most useful flags are summarised below.

| Script | Default | Important flags |
|---|---|---|
| `01_train.py` | 5 epochs, dropout 0.5, Adam lr 1e-3, **weight decay 1e-4** (matches paper) | `--epochs`, `--lr`, `--dropout`, `--weight-decay`, `--seed`, `--device {auto,cpu,cuda}` |
| `02_uncertainty.py` | T = 30 | `--T`, `--seed`, `--device` |
| `03_rotating_digit.py` | digit = **1**, T = 100, angles −90°…90° step 15° | `--digit`, `--T`, `--min-angle`, `--max-angle`, `--step`, `--index` (specific test image), `--seed` |
| `04_ood.py` | T = 30 | `--T`, `--seed`, `--device` |
| `05_accuracy_vs_T.py` | T sweep over {1, 2, 3, 5, 10, 20, 30, 50, 75, 100}, max-T = 100 | `--max-T` |
| `06_multiseed.py` | seeds 0 1 2, 3 epochs each, T = 30 | `--seeds 0 1 2`, `--epochs`, `--dropout`, `--T` |
| `07_uci_regression.py` | Concrete Strength, 5 random 90/10 splits, 400 epochs, p = 0.05, T = 1000, τ-grid | `--dataset {concrete,energy}`, `--n-splits`, `--epochs`, `--T`, `--p` |
| `08_extras.py` | uses the saved MNIST checkpoint, T = 30 for the OOD pass | (none) |

Experiments 2–6 and 8 all reuse the checkpoint saved by `01_train.py` (`results/model.pt`), so the model is only trained once.

The first run of `04_ood.py` and `07_uci_regression.py` downloads FashionMNIST and the OpenML Concrete dataset respectively into `data/`.

---

## The notebook (`reproduction.ipynb`)

`reproduction.ipynb` is the **polished, self-contained notebook** required by the assignment. It is *not* a shell wrapper — every cell imports from `src/mc_dropout`, trains the model in place, runs MC inference, and produces every figure inline. It has been executed end-to-end (all 13 code cells exit cleanly, outputs are embedded) so the embedded plots match what the LaTeX report shows.

Sections, in order:

0. Setup (imports, seed, device)
1. Data and model (architecture description)
2. Train (5 epochs, Adam, weight decay 1e-4)
3. Standard vs MC Dropout predictions
4. Uncertainty signals: entropy, variance, mutual information
5. Most-confident vs most-uncertain test images
6. **Rotating-digit signature experiment** (digit "1", T = 100), both views
7. **Out-of-distribution detection** on FashionMNIST (extension)
8. How many MC samples T are needed?
9. Multi-seed robustness (loaded from `results/multiseed_stats.csv`)
10. UCI regression Table 1 (loaded from `results/uci_regression.csv`)

The "loaded from CSV" sections in 9 and 10 use the artefacts of `06_multiseed.py` and `07_uci_regression.py` rather than retraining in the notebook (multi-seed is ~6 min, UCI is ~4 min; the rest of the notebook is ~10 min and is run inline).

To regenerate the notebook after editing the prose or code cells, edit `experiments/_build_notebook.py` and run:

```bash
python experiments/_build_notebook.py
jupyter nbconvert --to notebook --execute reproduction.ipynb --output reproduction.ipynb
```

---

## The report (`report.pdf` / `report.tex`)

`report.pdf` is the NeurIPS-2026-formatted writeup (preprint option, 9 pages total, body 5 pages + appendix 4 pages, single author).

Layout:

| Pages | Content |
|---|---|
| 1 | Title, abstract (with GitHub URL), Section 1 — paper summary, key equations |
| 2 | Section 1 continues; **Figure 1: rotating-digit per-pass scatter (paper Fig. 4 mirror)** |
| 3 | Section 2 — reproduced results, Table 1 (headline numbers vs paper) |
| 4 | Section 2 continues; Table 2 (UCI Concrete Strength vs paper); Section 3 — Discussion |
| 5 | References; Appendix A begins (architecture TikZ diagram) |
| 6 | Appendix figures: training, variance histogram, confident/uncertain examples |
| 7 | Rotating-digit mean view, OOD histogram, uncertainty decomposition |
| 8 | Wrong-confusion matrix, selective prediction, UCI scatter, reliability diagrams |
| 9 | Hyperparameter table, multi-seed table, UCI per-split table, T-sweep table, implementation notes, AI-use disclosure, work-distribution statement |

**Compiling locally.** Requires a LaTeX engine (MiKTeX, TeX Live, …) with the `neurips_2026.sty` file present (it lives next to `report.tex`). Two compile passes are needed to resolve cross-references:

```bash
pdflatex report.tex && pdflatex report.tex
```

If you don't have a local LaTeX install, you can also upload `report.tex`, `neurips_2026.sty`, and the `figures/` folder to [Overleaf](https://www.overleaf.com) and it will compile in the browser.

---

## Implementation notes

### MC Dropout in 6 lines

The whole stochastic-inference idea reduces to this helper (`src/mc_dropout/mcdropout.py`):

```python
def enable_dropout(model):
    """Put only the Dropout layers back into training mode."""
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.train()
```

Used as:

```python
model.eval()              # everything to eval (BN frozen, etc.)
enable_dropout(model)     # re-enable only dropout layers
probs_T = torch.stack([
    F.softmax(model(x), dim=1) for _ in range(T)
])                        # (T, B, C)
mean_probs = probs_T.mean(0)
```

### Uncertainty measures

From a tensor `probs_T` of shape `(T, N, C)`:

| Measure | Formula | What it captures |
|---|---|---|
| Predictive entropy | `H[E_t[p_t]] = − Σ_c p̄_c log p̄_c` | total (epistemic + aleatoric) |
| Expected entropy | `E_t[H[p_t]]` | aleatoric |
| Mutual information (BALD) | `H[E_t[p_t]] − E_t[H[p_t]]` | epistemic |
| Predictive variance | `Var_t[p_t]` averaged over classes | dispersion of T predictions |
| Negative log-likelihood | `−log p̄_{y_true}` | per-example fit |

All implemented in `src/mc_dropout/mcdropout.py` and independently verified on synthetic arrays (entropy ∈ [0, ln C], MI ≥ 0, MI = 0 when passes agree).

### Regression predictive log-likelihood (eq. 8)

The UCI regression test log-likelihood follows equation (8) of the paper directly:

```
log p(y*|x*, D)  ≈  logsumexp(−½ τ ‖y* − ŷ_t*‖²) − log T − ½ log(2π) − ½ log(τ⁻¹)
```

implemented in `src/mc_dropout/uci.py: mc_metrics`, with a numerically stable log-sum-exp. The model precision `τ` is chosen by an 8-point validation-set grid search — a cheaper substitute for the paper's Bayesian optimisation.

### Architecture

The classification network is a small modern CNN, not the exact LeNet the paper uses:

```
Conv(1, 32, 3×3) → ReLU → MaxPool(2)
   → Conv(32, 64, 3×3) → ReLU → MaxPool(2)
   → Flatten
   → Dropout(0.5) → FC(3136, 128) → ReLU
   → Dropout(0.5) → FC(128, 10)            # raw logits
```

422 k parameters. Dropout is placed before **both** fully-connected layers; either of these dropout layers being active at test time triggers MC Dropout.

The UCI regression model is the small MLP the paper itself uses for Table 1:

```
Dropout(0.05) → FC(8, 50) → ReLU → Dropout(0.05) → FC(50, 1)
```

---

## Reproducibility

- **Seeds.** Every script seeds Python, NumPy and PyTorch from a single `--seed` flag (default 42). The multi-seed experiment varies it explicitly across `--seeds 0 1 2`.
- **Pinned dependencies.** `requirements.txt` lists direct dependencies with conservative lower bounds:

  ```
  torch>=2.0.0
  torchvision>=0.15.0
  numpy>=1.23.0
  pandas>=1.5.0
  matplotlib>=3.6.0
  scikit-learn>=1.2.0
  tqdm>=4.65.0
  ```

- **Windows note.** PyTorch DataLoader's `num_workers > 0` can crash on Windows on some configurations; the loaders here default to `num_workers=0` for safety.
- **No GPU required.** Everything runs on CPU; total end-to-end ≈ 30 minutes including multi-seed and UCI.
- **Determinism.** Results vary slightly between runs because the MC inference samples dropout masks; the multi-seed CSV gives a sense of the variance.

---

## Citation

If you reuse this code, please cite the original paper:

```bibtex
@inproceedings{gal2016dropout,
  title     = {Dropout as a {B}ayesian Approximation: Representing Model Uncertainty in Deep Learning},
  author    = {Gal, Yarin and Ghahramani, Zoubin},
  booktitle = {ICML},
  year      = {2016}
}
```

## License

Released for academic coursework use.
