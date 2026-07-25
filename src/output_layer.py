from typing import Callable
import numpy as np
from layer import Layer


class Output_Layer(Layer):
    """Represent the final layer of the network."""

    def __init__(self, node_count: int,
                 activation: Callable[[float], float],
                 activation_Derivative: Callable[[float], float]) -> None:
        """Initialize an output layer with the requested number of nodes.

        Args:
            node_count: Number of nodes to create in the layer.
            activation: Activation function applied by each node.
            activation_Derivative: Derivative of the activation function.
        """
        super().__init__(node_count, activation, activation_Derivative)

    def calc_output_gradients(self, error_derivatives: np.ndarray) -> np.ndarray:
        """Return the gradients for each output node.
        
        Args:
            error_derivatives: Error derivatives for each output node.

        Returns:
            Element-wise gradients for the output layer.
        """
        self.dels = np.array([node.activation_prime(node.weighted_sum) for node in self.nodes]) * error_derivatives
        return self.dels