"""Single-neuron calculations for the multilayer perceptron.

Each Node stores a scalar activation function and its derivative, then applies the
bias-adjusted weighted sum used during forward propagation and backpropagation.
"""

import numpy as np
from typing import Callable


class Node:
    """Represent a single artificial neuron with an activation function."""

    activation: Callable[[float], float]
    activation_prime: Callable[[float], float]
    weighted_sum: float

    def __init__(self, activation_function: Callable[[float], float], activation_prime: Callable[[float], float]) -> None:
        """Store the activation and derivative callbacks for the node.

        Args:
            activation_function: Activation function applied to the node's weighted sum.
            activation_prime: Derivative of the activation function.
        """
        self.activation = activation_function
        self.activation_prime = activation_prime

    def calculate(self, inputs: np.ndarray, input_weights: np.ndarray) -> float:
        """Compute the node output for a given input vector and store its weighted sum.

        Args:
            inputs: Input values passed to the node.
            input_weights: Bias and feature weights; must have length len(inputs) + 1.

        Returns:
            The activated scalar output of the node.

        Raises:
            ValueError: If the weight vector length does not include the bias term.
        """
        if len(input_weights) != len(inputs) + 1:
            raise ValueError("Weight array must be of length one greater than input array")

        # The first weight acts as the bias term; remaining weights scale each input feature.
        self.weighted_sum = input_weights[0] + np.dot(inputs, input_weights[1:])
        return self.activation(self.weighted_sum)

    def calc_gradient(self, output_weights: np.ndarray, output_dels: np.ndarray) -> float:
        """Return the gradient of this node's local error contribution.

        Args:
            output_weights: Weights connecting this node to the downstream layer.
            output_dels: Error derivatives from the downstream layer.

        Returns:
            The gradient for this node's weighted sum.
        """
        # Chain rule: local derivative = activation derivative × weighted sum of downstream deltas.
        return self.activation_prime(self.weighted_sum) * np.dot(output_weights, output_dels)