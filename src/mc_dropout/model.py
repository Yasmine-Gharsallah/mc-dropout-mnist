"""The convolutional network used throughout the project.

The dropout layers placed before the two fully-connected layers are the ones we
keep active at test time to perform MC Dropout.
"""

import torch.nn as nn
import torch.nn.functional as F


class MCDropoutCNN(nn.Module):
    """A small CNN for MNIST with two dropout layers before the FC layers."""

    def __init__(self, p: float = 0.5, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.drop1 = nn.Dropout(p)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.drop2 = nn.Dropout(p)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))   # 28 -> 14
        x = self.pool(F.relu(self.conv2(x)))   # 14 -> 7
        x = x.flatten(1)
        x = self.drop1(x)
        x = F.relu(self.fc1(x))
        x = self.drop2(x)
        return self.fc2(x)                     # raw logits
