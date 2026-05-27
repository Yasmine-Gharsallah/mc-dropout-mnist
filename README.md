# MC Dropout for Uncertainty Estimation on MNIST

Reimplementation of the practical core of:

> Yarin Gal and Zoubin Ghahramani. *Dropout as a Bayesian Approximation: Representing
> Model Uncertainty in Deep Learning.* ICML 2016. [arXiv:1506.02142](https://arxiv.org/abs/1506.02142)

The paper shows that a neural network trained with dropout is, mathematically, an
approximation to a Bayesian model (variational inference in a deep Gaussian process).
The practical consequence is **Monte Carlo (MC) Dropout**: keep dropout *active at test
time*, run `T` stochastic forward passes per input, and use the spread of the predictions
as a principled estimate of model uncertainty.

This repository reproduces that practical idea on MNIST and studies where the resulting
uncertainty behaves as the theory predicts.

## What this project does

1. Trains a small CNN with dropout on MNIST.
2. Evaluates it the **standard** way (`model.eval()`, dropout off) and with **MC Dropout**
   (dropout on, `T` stochastic passes), and compares accuracy.
3. Computes three uncertainty measures from the `T` passes: **predictive entropy**,
   **predictive variance**, and **mutual information (BALD)**.
4. Checks that **wrong / ambiguous** predictions receive higher uncertainty than confident
   correct ones.
5. Reproduces the paper's signature **rotating-digit** experiment: uncertainty grows as a
   digit is rotated into an ambiguous shape.
6. Tests **out-of-distribution (OOD)** behaviour: uncertainty on FashionMNIST (which the
   model was never trained on) versus in-distribution MNIST, scored with AUROC.
7. Measures how the MC estimate **converges with the number of samples `T`** (accuracy and
   negative log-likelihood vs `T`).
8. Repeats the headline experiment over **several random seeds** and reports mean ± std.

## Repository structure

```
.
├── README.md
├── requirements.txt
├── notebook.ipynb              # end-to-end runner (Colab-friendly, good for figures)
├── src/mc_dropout/             # reusable library code
│   ├── __init__.py
│   ├── utils.py                # seeding, device
│   ├── model.py                # MCDropoutCNN
│   ├── data.py                 # MNIST loaders, rotation, OOD (FashionMNIST)
│   ├── mcdropout.py            # enable_dropout, standard/MC prediction, uncertainty metrics
│   └── train.py                # training loop, checkpointing
├── experiments/                # one script per experiment, each saves figures + CSVs
│   ├── 01_train.py
│   ├── 02_uncertainty.py
│   ├── 03_rotating_digit.py
│   ├── 04_ood.py
│   ├── 05_accuracy_vs_T.py
│   └── 06_multiseed.py
├── figures/                    # generated plots
└── results/                    # generated CSV summaries
```

## How to run

### Local

```bash
pip install -r requirements.txt

# train once (saves results/model.pt and figures/training_curves.png)
python experiments/01_train.py

# then run the analyses (each reuses the saved checkpoint)
python experiments/02_uncertainty.py
python experiments/03_rotating_digit.py
python experiments/04_ood.py
python experiments/05_accuracy_vs_T.py

# multi-seed statistics (retrains a few small models)
python experiments/06_multiseed.py
```

All scripts accept `--help` for options (epochs, dropout rate, number of MC samples `T`,
seed, device, …).

### Google Colab

Open `notebook.ipynb`, set the runtime to GPU, and run all cells. Training takes ~1–2
minutes on a T4.

## Results

The generated figures land in `figures/` and the numeric summaries in `results/`.
Expected qualitative findings (your exact numbers will vary slightly by seed/hardware):

- Standard and MC-Dropout test accuracy are essentially identical (MC Dropout is not an
  accuracy booster on clean MNIST); MC Dropout's value is the **uncertainty** it adds.
- Wrong predictions carry **much higher** predictive entropy than correct ones.
- The most uncertain test images are visually ambiguous digits; the most confident are
  clean, prototypical ones.
- Rotating a digit drives the predictive entropy up as the shape becomes ambiguous.
- FashionMNIST inputs receive systematically higher entropy than MNIST, giving strong
  OOD-detection AUROC using uncertainty alone.
- The MC estimate stabilises after roughly 20–50 forward passes.

## License

Released for academic coursework use.
