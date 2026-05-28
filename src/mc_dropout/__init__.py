"""MC Dropout on MNIST: reusable library code.

Reimplementation of the practical core of Gal & Ghahramani (2016),
"Dropout as a Bayesian Approximation".
"""

from .utils import set_seed, get_device
from .model import MCDropoutCNN
from .data import (
    get_mnist_loaders,
    get_raw_test_dataset,
    get_ood_loader,
    rotate_and_normalise,
    normalise,
)
from .mcdropout import (
    enable_dropout,
    standard_predict,
    mc_forward,
    mc_forward_batch,
    predictive_mean,
    predictive_entropy,
    expected_entropy,
    mutual_information,
    predictive_variance,
    negative_log_likelihood,
    summarise,
)
from .train import train_model, save_checkpoint, load_checkpoint
from .uci import DropoutMLP, train_mlp, mc_predict, mc_metrics, search_tau

__all__ = [
    "set_seed", "get_device",
    "MCDropoutCNN",
    "get_mnist_loaders", "get_raw_test_dataset", "get_ood_loader",
    "rotate_and_normalise", "normalise",
    "enable_dropout", "standard_predict", "mc_forward", "mc_forward_batch",
    "predictive_mean", "predictive_entropy", "expected_entropy",
    "mutual_information", "predictive_variance", "negative_log_likelihood",
    "summarise",
    "train_model", "save_checkpoint", "load_checkpoint",
    "DropoutMLP", "train_mlp", "mc_predict", "mc_metrics", "search_tau",
]
