"""TensorFlow 2 / Keras implementation of a classic 2-layer GCN.

Model architecture:
    GCNLayer(hidden, relu) -> Dropout -> GCNLayer(num_classes) -> log_softmax

The graph convolution is implemented as a Keras custom layer:
    output = (A_hat @ X) @ W + b
where ``A_hat`` is the symmetric-normalized adjacency matrix (with self-loops).
The module does not depend on any third-party GNN library.
"""

from __future__ import annotations

import tensorflow as tf
from tensorflow import keras


class GCNLayer(keras.layers.Layer):
    """Single graph convolutional layer.

    It expects inputs as a tuple/list ``(x, adjacency)`` where:

    * ``x`` has shape ``[num_nodes, in_features]``.
    * ``adjacency`` is the normalized adjacency matrix with shape
      ``[num_nodes, num_nodes]``, either dense or ``tf.SparseTensor``.

    The forward pass computes ``activation((A_hat @ X) @ W + b)``.
    """

    def __init__(self, units: int, activation=None, use_bias: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.units = int(units)
        self.activation = keras.activations.get(activation)
        self.use_bias = bool(use_bias)

    def build(self, input_shape):
        # ``input_shape`` is a tuple of shapes: (feature_shape, adjacency_shape).
        feature_shape = input_shape[0]
        in_features = int(feature_shape[-1])

        self.kernel = self.add_weight(
            name="kernel",
            shape=(in_features, self.units),
            initializer="glorot_uniform",
            trainable=True,
        )

        if self.use_bias:
            self.bias = self.add_weight(
                name="bias",
                shape=(self.units,),
                initializer="zeros",
                trainable=True,
            )
        else:
            self.bias = None

        super().build(input_shape)

    def call(self, inputs, training=None):
        del training  # GCNLayer has no training-specific behavior.
        x, adjacency = inputs

        # Message passing / feature propagation: A_hat @ X.
        if isinstance(adjacency, tf.SparseTensor):
            support = tf.sparse.sparse_dense_matmul(adjacency, x)
        else:
            support = tf.matmul(adjacency, x)

        # Linear transformation: (A_hat @ X) @ W + b.
        output = support @ self.kernel
        if self.use_bias:
            output = output + self.bias

        if self.activation is not None:
            output = self.activation(output)

        return output


class GCN(keras.Model):
    """Classic 2-layer GCN for node classification.

    Architecture:
        GCNLayer(hidden_channels, relu)
        -> Dropout(dropout)
        -> GCNLayer(num_classes)
        -> log_softmax
    """

    def __init__(self, num_classes: int, hidden_channels: int = 16, dropout: float = 0.5, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = int(num_classes)
        self.hidden_channels = int(hidden_channels)
        self.dropout_rate = float(dropout)

        self.conv1 = GCNLayer(self.hidden_channels, activation="relu")
        self.dropout = keras.layers.Dropout(self.dropout_rate)
        self.conv2 = GCNLayer(self.num_classes)

    def call(self, inputs, training=False):
        """Forward pass.

        Args:
            inputs: Tuple/list ``(x, adjacency)``.
            training: Whether the model is in training mode (controls dropout).

        Returns:
            Log-probabilities with shape ``[num_nodes, num_classes]``.
        """
        x, adjacency = inputs

        x = self.conv1((x, adjacency))
        x = self.dropout(x, training=training)
        x = self.conv2((x, adjacency))

        return tf.nn.log_softmax(x, axis=-1)
