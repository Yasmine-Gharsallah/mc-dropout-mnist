"""Data loading: MNIST train/test loaders, single-image rotation for the
rotating-digit experiment, and a FashionMNIST loader used as out-of-distribution
(OOD) data.

The OOD data is deliberately normalised with the *MNIST* statistics, because at
test time the model only knows the MNIST preprocessing it was trained with.
"""

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode

# Standard MNIST normalisation constants.
MNIST_MEAN = 0.1307
MNIST_STD = 0.3081


def get_transform():
    """ToTensor + MNIST normalisation."""
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((MNIST_MEAN,), (MNIST_STD,)),
    ])


def get_mnist_loaders(batch_size=128, data_dir="./data", num_workers=0, download=True):
    """Return (train_loader, test_loader) for MNIST with standard normalisation."""
    transform = get_transform()
    train_set = datasets.MNIST(data_dir, train=True, download=download, transform=transform)
    test_set = datasets.MNIST(data_dir, train=False, download=download, transform=transform)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers)
    return train_loader, test_loader


def get_raw_test_dataset(data_dir="./data", download=True):
    """MNIST test set with pixels in [0, 1] (no normalisation).

    Used by the rotating-digit experiment so we can rotate the raw image on a
    black background and only afterwards apply normalisation.
    """
    return datasets.MNIST(data_dir, train=False, download=download,
                           transform=transforms.ToTensor())


def normalise(img):
    """Apply MNIST normalisation to a [0, 1] tensor of shape (1, H, W)."""
    return TF.normalize(img, [MNIST_MEAN], [MNIST_STD])


def rotate_and_normalise(raw_img, angle):
    """Rotate a raw [0, 1] image by `angle` degrees (black fill) and normalise.

    raw_img: tensor of shape (1, 28, 28) with values in [0, 1].
    Returns a normalised tensor of shape (1, 28, 28).
    """
    rotated = TF.rotate(raw_img, angle, interpolation=InterpolationMode.BILINEAR, fill=0.0)
    return normalise(rotated)


def get_ood_loader(batch_size=128, data_dir="./data", num_workers=0, download=True):
    """FashionMNIST test set, normalised with MNIST statistics, as OOD data."""
    transform = get_transform()
    ood_set = datasets.FashionMNIST(data_dir, train=False, download=download,
                                    transform=transform)
    return DataLoader(ood_set, batch_size=batch_size, shuffle=False,
                      num_workers=num_workers)
