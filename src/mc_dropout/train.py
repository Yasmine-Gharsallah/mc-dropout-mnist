"""Training loop and checkpoint helpers."""

import torch
import torch.nn as nn
from tqdm.auto import tqdm


def train_model(model, train_loader, epochs=5, lr=1e-3, device="cpu",
                weight_decay=0.0, verbose=True):
    """Train `model` with Adam + cross-entropy. Returns a history dict.

    `weight_decay` corresponds to the L2 regularisation that, in the paper's
    interpretation, pairs with dropout to make training approximate variational
    inference. It defaults to 0 to keep the baseline simple.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    history = {"train_loss": [], "train_acc": []}

    for epoch in range(1, epochs + 1):
        model.train()
        total, correct, loss_sum = 0, 0, 0.0
        iterator = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}") if verbose else train_loader
        for x, y in iterator:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            loss_sum += loss.item() * x.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            total += x.size(0)

        train_loss = loss_sum / total
        train_acc = correct / total
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        if verbose:
            print(f"Epoch {epoch}: loss={train_loss:.4f}  acc={train_acc:.4f}")

    return history


def save_checkpoint(model, path, meta=None):
    """Save model weights (and optional metadata dict) to `path`."""
    torch.save({"state_dict": model.state_dict(), "meta": meta or {}}, path)


def load_checkpoint(model, path, device="cpu"):
    """Load weights into `model` from `path` and return the metadata dict."""
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    return ckpt.get("meta", {})
