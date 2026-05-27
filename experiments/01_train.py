"""Experiment 1 — train the CNN with dropout on MNIST.

Saves the trained weights to results/model.pt and the training curves to
figures/training_curves.png. All later experiments reuse this checkpoint.
"""

import argparse

import common  # noqa: F401  (sets up import path + output dirs)
import matplotlib.pyplot as plt

from mc_dropout import (
    set_seed, get_device, MCDropoutCNN, get_mnist_loaders,
    train_model, save_checkpoint,
)


def main():
    parser = argparse.ArgumentParser(description="Train MCDropoutCNN on MNIST")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device(args.device)
    print("Using device:", device)

    train_loader, _ = get_mnist_loaders(batch_size=args.batch_size, data_dir=common.DATA_DIR)

    model = MCDropoutCNN(p=args.dropout).to(device)
    print(model)
    print("Trainable parameters:", sum(p.numel() for p in model.parameters()))

    history = train_model(
        model, train_loader, epochs=args.epochs, lr=args.lr,
        device=device, weight_decay=args.weight_decay,
    )

    meta = {"dropout": args.dropout, "epochs": args.epochs, "seed": args.seed}
    save_checkpoint(model, common.MODEL_PATH, meta=meta)
    print("Saved checkpoint to", common.MODEL_PATH)

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
    ax[0].plot(range(1, args.epochs + 1), history["train_loss"], marker="o")
    ax[0].set_title("Training loss"); ax[0].set_xlabel("epoch"); ax[0].set_ylabel("loss")
    ax[1].plot(range(1, args.epochs + 1), history["train_acc"], marker="o", color="tab:green")
    ax[1].set_title("Training accuracy"); ax[1].set_xlabel("epoch"); ax[1].set_ylabel("accuracy")
    fig.tight_layout()
    out = f"{common.FIG_DIR}/training_curves.png"
    fig.savefig(out, dpi=150)
    print("Saved", out)


if __name__ == "__main__":
    main()
