from node import Node
from typing import Callable
import numpy as np

class Layer:
    """Represent a collection of nodes that share the same activation function."""

    nodes: list[Node]

    def __init__(self, node_count: int,
                 activation: Callable[[float], float],
                 activation_Derivative: Callable[[float], float]) -> None:
        """Initialize a layer with a given number of nodes.

        Args:
            node_count: Number of nodes to create in the layer.
            activation: Activation function applied by each node.
            activation_Derivative: Derivative of the activation function.
        """
        self.nodes = [Node(activation, activation_Derivative) for i in range(node_count)]

    def calculate(self, inputs: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """Compute the output of every node in the layer for the provided inputs.

        Args:
            inputs: Input array passed to each node.
            weights: Weight array used by each node to compute its weighted sum.

        Returns:
            Array of outputs from each node in the layer.
        """
        values = [node.calculate(inputs, weights) for node in self.nodes]
        return np.array(values)

    def calc_gradients(self, output_weights: np.ndarray, output_dels: np.ndarray) -> np.ndarray:
        """Compute the gradient for each node in the layer.

        Args:
            output_weights: Weight array connecting this layer to the next layer.
            output_dels: Error derivatives for the downstream layer.

        Returns:
            Array of gradients for each node in the layer.
        """
        dels = [node.calc_gradient(output_weights, output_dels) for node in self.nodes]
        return np.array(dels)