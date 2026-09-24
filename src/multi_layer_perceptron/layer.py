"""Layer and node grouping utilities for the multilayer perceptron.

A layer represents a collection of neurons that share the same activation and
activation-derivative functions.
"""

from .node import Node
from typing import Callable
import numpy as np


class Layer:
    """Represent a collection of nodes that share the same activation function."""

    nodes: list[Node]
    outputs: np.ndarray
    dels: np.ndarray

    def __init__(self, node_count: int,
                 activation: Callable[[float], float],
                 activation_Derivative: Callable[[float], float]) -> None:
        """Create a layer with the requested number of nodes.

        Args:
            node_count: Number of nodes to create in the layer.
            activation: Activation function applied by each node.
            activation_Derivative: Derivative of the activation function.
        """
        self.nodes = [Node(activation, activation_Derivative) for _ in range(node_count)]

    def calculate(self, inputs: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """Compute the output of every node in the layer for the provided inputs.

        Args:
            inputs: Input array passed to each node.
            weights: Weight array used by each node to compute its weighted sum.

        Returns:
            An array of outputs from each node in the layer.
        """
        inputs = np.asarray(inputs, dtype=float)
        weights = np.asarray(weights, dtype=float)

        if inputs.ndim != 1:
            raise ValueError("Input array must be a 1D vector")

        if weights.ndim != 2:
            raise ValueError("Weight array must be a 2D matrix with one column per node")

        if weights.shape[0] != len(inputs) + 1:
            raise ValueError("Weight array must have one more row than the input vector length")

        if weights.shape[1] != len(self.nodes):
            raise ValueError("Weight array must have one column per node in the layer")

        # Each node uses the same input vector but a different column of the weight matrix.
        self.outputs = np.array([node.calculate(inputs, weights[:, i]) for i, node in enumerate(self.nodes)])
        return np.array(self.outputs)

    def calc_gradients(self, output_weights: np.ndarray, output_dels: np.ndarray) -> np.ndarray:
        """Compute the gradient for each node in the layer.

        Args:
            output_weights: Weight array connecting this layer to the next layer.
            output_dels: Error derivatives for the downstream layer.

        Returns:
            An array of gradients for each node in the layer.
        """
        # The bias row is skipped when reading the downstream weights for each node.
        self.dels = np.array([node.calc_gradient(output_weights[i + 1, :], output_dels) for i, node in enumerate(self.nodes)])
        return self.dels

    def node_count(self) -> int:
        """Return the number of nodes in the layer."""
        return len(self.nodes)