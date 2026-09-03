"""Command-line entry point for the GCN node classifier.

Run with:

    python -m gnn_node_classifier [--epochs 200] [--hidden 16] [--lr 0.01]
                                   [--seed 42] [--data-dir data/cora]
"""

from __future__ import annotations

import argparse

import numpy as np

from .data import load_cora
from .train import train_model


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {value!r}")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"expected a positive number, got {value!r}")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gnn_node_classifier",
        description="Train a 2-layer GCN for node classification on the Cora dataset.",
    )
    parser.add_argument("--epochs", type=_positive_int, default=200,
                        help="number of full-batch training epochs (default: 200)")
    parser.add_argument("--hidden", type=_positive_int, default=16,
                        help="hidden layer size (default: 16)")
    parser.add_argument("--lr", type=_positive_float, default=0.01,
                        help="learning rate for Adam (default: 0.01)")
    parser.add_argument("--weight-decay", type=float, default=5e-4,
                        help="L2 regularization coefficient for kernel weights (default: 5e-4)")
    parser.add_argument("--seed", type=int, default=42,
                        help="random seed (default: 42)")
    parser.add_argument("--data-dir", type=str, default="data/cora",
                        help="directory containing or receiving cora.content and cora.cites")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    if args.weight_decay < 0.0:
        raise SystemExit("--weight-decay must be non-negative")

    print("=" * 60)
    print("GCN Node Classifier (TensorFlow 2 + Keras)")
    print("=" * 60)

    print("\nLoading Cora dataset ...")
    data = load_cora(data_dir=args.data_dir, seed=args.seed)

    print("\nDataset statistics:")
    print(f"  nodes          : {data['num_nodes']}")
    print(f"  edges (undir.) : {data['num_edges']}")
    print(f"  features       : {data['num_features']}")
    print(f"  classes        : {data['num_classes']}")
    print(f"  train mask     : {int(np.sum(data['train_mask']))}")
    print(f"  val mask       : {int(np.sum(data['val_mask']))}")
    print(f"  test mask      : {int(np.sum(data['test_mask']))}")
    print(f"  unlabeled      : {int(np.sum(~(data['train_mask'] | data['val_mask'] | data['test_mask'])))}")

    print(
        f"\nTraining GCN with hidden={args.hidden}, "
        f"lr={args.lr}, weight_decay={args.weight_decay}, epochs={args.epochs} ..."
    )

    model, metrics = train_model(
        data,
        hidden_channels=args.hidden,
        lr=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        seed=args.seed,
    )

    print("\nTraining finished.")
    print("-" * 60)
    print(f"Final train accuracy : {metrics['final_train_acc']:.4f}")
    print(f"Final val accuracy   : {metrics['final_val_acc']:.4f}")
    print(f"Final test accuracy  : {metrics['final_test_acc']:.4f}")
    print("-" * 60)
    print(f"Best epoch           : {metrics['best_epoch']}")
    print(f"Best val accuracy    : {metrics['best_val_acc']:.4f}")
    print(f"Best test accuracy   : {metrics['best_test_acc']:.4f}")
    print("-" * 60)

    del model  # The metrics dict is the CLI's primary result.


if __name__ == "__main__":
    main()
