"""Shared setup for the experiment scripts: import path, output folders, paths.

Importing this module makes the `mc_dropout` package importable and creates the
`figures/` and `results/` directories. It also forces a non-interactive
matplotlib backend so the scripts run headless.
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
SRC = os.path.join(REPO_ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

FIG_DIR = os.path.join(REPO_ROOT, "figures")
RES_DIR = os.path.join(REPO_ROOT, "results")
DATA_DIR = os.path.join(REPO_ROOT, "data")
MODEL_PATH = os.path.join(RES_DIR, "model.pt")

os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(RES_DIR, exist_ok=True)
