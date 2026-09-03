"""Unit tests for the TensorFlow 2 + Keras GCN node classifier.

These tests are fully offline: they use a small synthetic 5-node graph and do
not download Cora. They are skipped when TensorFlow (or another required GNN
dependency) is unavailable.
"""

from __future__ import annotations

import math
import unittest

import numpy as np

try:
    import tensorflow as tf

    from gnn_node_classifier import GCN, evaluate, train_one_epoch
    from gnn_node_classifier.data import normalize_adjacency

    GNN_AVAILABLE = True
except ImportError:  # pragma: no cover - depends on the local environment.
    GNN_AVAILABLE = False


def _make_synthetic_data():
    """Create a tiny 5-node, 8-feature, 7-class synthetic graph."""
    rng = np.random.RandomState(0)
    num_nodes = 5
    num_features = 8
    num_classes = 7

    features = rng.rand(num_nodes, num_features).astype(np.float32)
    labels = np.array([0, 1, 2, 3, 4], dtype=np.int64)

    # Symmetric normalized adjacency for a small connected graph with self-loops.
    # A dense numpy adjacency is used here so the model can be called directly;
    # train_one_epoch/evaluate also accept dense tensors.
    edge_index = np.array([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=np.int64)
    adjacency = normalize_adjacency(edge_index, num_nodes).toarray().astype(np.float32)

    masks = np.ones(num_nodes, dtype=bool)
    return {
        "features": features,
        "adjacency": adjacency,
        "labels": labels,
        "num_classes": num_classes,
        "train_mask": masks,
        "val_mask": masks,
        "test_mask": masks,
    }


@unittest.skipUnless(GNN_AVAILABLE, "TensorFlow is not installed; skipping GNN tests.")
class GNNNodeClassifierTests(unittest.TestCase):
    def setUp(self) -> None:
        tf.random.set_seed(0)
        self.data = _make_synthetic_data()

    def test_forward_output_shape(self) -> None:
        model = GCN(num_classes=7)
        log_probs = model((self.data["features"], self.data["adjacency"]), training=False)

        self.assertEqual(log_probs.shape.as_list(), [5, 7])

    def test_output_is_log_probability(self) -> None:
        model = GCN(num_classes=7)
        log_probs = model((self.data["features"], self.data["adjacency"]), training=False)

        probabilities = tf.reduce_sum(tf.exp(log_probs), axis=1)
        np.testing.assert_allclose(probabilities.numpy(), np.ones(5, dtype=np.float32), atol=1e-5)

    def test_train_step_loss_is_finite(self) -> None:
        model = GCN(num_classes=7)
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

        losses = []
        for _ in range(3):
            loss = train_one_epoch(model, self.data, optimizer, loss_fn)
            losses.append(loss)

        for loss in losses:
            self.assertTrue(math.isfinite(loss), f"loss is not finite: {loss!r}")
            self.assertFalse(math.isnan(loss))

    def test_evaluate_returns_accuracy_range(self) -> None:
        model = GCN(num_classes=7)
        train_acc, val_acc, test_acc = evaluate(model, self.data)

        for accuracy in (train_acc, val_acc, test_acc):
            self.assertGreaterEqual(accuracy, 0.0)
            self.assertLessEqual(accuracy, 1.0)

    def test_normalize_adjacency_shape_and_nonzero_rows(self) -> None:
        edge_index = np.array([[0, 1, 2], [1, 2, 0]], dtype=np.int64)
        adjacency = normalize_adjacency(edge_index, num_nodes=5)

        self.assertEqual(adjacency.shape, (5, 5))

        row_sums = np.asarray(adjacency.sum(axis=1)).ravel()
        self.assertTrue(np.all(row_sums > 0), "self-loops should leave no isolated node")


if __name__ == "__main__":
    unittest.main()
