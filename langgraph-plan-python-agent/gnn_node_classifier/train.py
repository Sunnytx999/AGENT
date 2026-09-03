"""Training and evaluation helpers for the 2-layer GCN node classifier."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import tensorflow as tf

from .model import GCN


def _to_feature_tensor(features) -> tf.Tensor:
    """Convert node features to a float32 ``tf.Tensor``."""
    if isinstance(features, tf.Tensor):
        return tf.cast(features, tf.float32)
    return tf.convert_to_tensor(features, dtype=tf.float32)


def _to_adjacency_tensor(adjacency):
    """Convert a dense or sparse adjacency to a TensorFlow tensor.

    ``scipy.sparse`` matrices are converted to ``tf.SparseTensor`` (memory
    efficient for Cora). NumPy arrays, ``tf.Tensor`` and ``tf.SparseTensor``
    are passed through/coerced appropriately.
    """
    if isinstance(adjacency, tf.SparseTensor):
        return adjacency
    if sp.issparse(adjacency):
        adjacency = adjacency.tocoo()
        indices = np.stack([adjacency.row, adjacency.col], axis=1).astype(np.int64)
        values = adjacency.data.astype(np.float32)
        sparse_tensor = tf.SparseTensor(
            indices=indices,
            values=values,
            dense_shape=np.asarray(adjacency.shape, dtype=np.int64),
        )
        return tf.sparse.reorder(sparse_tensor)
    if isinstance(adjacency, tf.Tensor):
        return tf.cast(adjacency, tf.float32)
    return tf.convert_to_tensor(adjacency, dtype=tf.float32)


def train_one_epoch(model: GCN, data: dict, optimizer, loss_fn, weight_decay: float = 0.0) -> float:
    """Perform one full-batch training step on the masked training nodes.

    Args:
        model: A ``GCN`` instance (or any model accepting ``(x, adjacency)``).
        data: Dict with ``features``, ``adjacency``, ``labels`` and
            ``train_mask``.
        optimizer: Keras/TensorFlow optimizer.
        loss_fn: Loss callable, e.g. ``SparseCategoricalCrossentropy``.
        weight_decay: Optional L2 penalty applied to kernel weights.

    Returns:
        The scalar loss value (float).
    """
    features = _to_feature_tensor(data["features"])
    adjacency = _to_adjacency_tensor(data["adjacency"])
    labels = tf.convert_to_tensor(data["labels"], dtype=tf.int64)
    train_mask = tf.convert_to_tensor(data["train_mask"], dtype=tf.bool)

    with tf.GradientTape() as tape:
        logits = model((features, adjacency), training=True)
        loss = loss_fn(tf.boolean_mask(labels, train_mask), tf.boolean_mask(logits, train_mask))

        if weight_decay > 0.0:
            l2_penalty = tf.add_n(
                [tf.nn.l2_loss(var) for var in model.trainable_variables if "kernel" in var.name]
            )
            loss = loss + weight_decay * l2_penalty

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return float(loss.numpy())


def evaluate(model: GCN, data: dict) -> tuple[float, float, float]:
    """Evaluate the model on the train/validation/test masks.

    Args:
        model: A ``GCN`` instance.
        data: Dict with ``features``, ``adjacency``, ``labels`` and the three
            boolean masks.

    Returns:
        Tuple ``(train_acc, val_acc, test_acc)``.
    """
    features = _to_feature_tensor(data["features"])
    adjacency = _to_adjacency_tensor(data["adjacency"])
    labels = tf.convert_to_tensor(data["labels"], dtype=tf.int64)

    logits = model((features, adjacency), training=False)
    predictions = tf.argmax(logits, axis=-1)

    accuracies = []
    for mask_name in ("train_mask", "val_mask", "test_mask"):
        mask = tf.convert_to_tensor(data[mask_name], dtype=tf.bool)
        masked_labels = tf.boolean_mask(labels, mask)
        masked_predictions = tf.boolean_mask(predictions, mask)
        correct = tf.reduce_sum(tf.cast(tf.equal(masked_labels, masked_predictions), tf.float32))
        accuracy = correct / tf.cast(tf.size(masked_labels), tf.float32)
        accuracies.append(float(accuracy.numpy()))

    return tuple(accuracies)  # type: ignore[return-value]


def train_model(
    data: dict,
    *,
    hidden_channels: int = 16,
    lr: float = 0.01,
    weight_decay: float = 5e-4,
    epochs: int = 200,
    seed: int = 42,
) -> tuple[GCN, dict]:
    """Train the 2-layer GCN and return the model plus training metrics.

    The test accuracy reported in ``best_test_acc`` comes from the epoch with
    the best validation accuracy, avoiding test-set leakage.

    Args:
        data: Dict returned by ``load_cora`` (or a synthetic equivalent).
        hidden_channels: Size of the hidden GCN layer.
        lr: Learning rate for Adam.
        weight_decay: L2 regularization coefficient for kernel weights.
        epochs: Number of full-batch training epochs.
        seed: Random seed for reproducible initialization.

    Returns:
        Tuple ``(model, metrics)`` where ``metrics`` contains histories and
        ``best_val_acc``/``best_test_acc``/``final_*`` values.
    """
    tf.random.set_seed(seed)
    np.random.seed(seed)

    num_classes = int(data["num_classes"])
    model = GCN(num_classes=num_classes, hidden_channels=hidden_channels)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    train_losses: list[float] = []
    train_accs: list[float] = []
    val_accs: list[float] = []
    test_accs: list[float] = []

    best_val_acc = -1.0
    best_test_acc = 0.0
    best_epoch = 0

    for epoch in range(1, epochs + 1):
        loss = train_one_epoch(model, data, optimizer, loss_fn, weight_decay=weight_decay)
        train_acc, val_acc, test_acc = evaluate(model, data)

        train_losses.append(loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        test_accs.append(test_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc
            best_epoch = epoch

    metrics = {
        "history": {
            "train_loss": train_losses,
            "train_acc": train_accs,
            "val_acc": val_accs,
            "test_acc": test_accs,
        },
        "best_val_acc": best_val_acc,
        "best_test_acc": best_test_acc,
        "best_epoch": best_epoch,
        "final_train_acc": train_accs[-1],
        "final_val_acc": val_accs[-1],
        "final_test_acc": test_accs[-1],
    }

    return model, metrics
