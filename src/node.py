import numpy as np
from typing import Callable

class Perceptron_Node:

    def __init__(self, input_count: int, weights: np.ndarray[tuple[int], np.dtype[np.float64]], activation_function: Callable[[np.float64], np.float64]) -> None:
        """Initialize a node with weights and an activation function.

        Args:
            input_count: Number of expected input values.
            weights: Weight vector including bias at index 0. Length must equal input_count + 1.
            activation_function: Activation function applied to the node's weighted sum.

        Raises:
            ValueError: If len(weights) != input_count + 1.
        """
        
        self.input_count = input_count
        # size of w must be equal to the number of inputs (plus one for bias)
        if len(weights) != input_count + 1:
            raise ValueError("Length of weights must be one more than the number of inputs")
        self.weights = weights
        self.phi = activation_function

    def set_weights(self, weights: np.ndarray) -> None:
        self.weights = weights

    def calculate(self, inputs: np.ndarray[tuple[int], np.dtype[np.float64]]) -> np.float64:
        """Compute the node output for a given input vector.

        Args:
            inputs: Input array of shape (input_count,).

        Returns:
            Activated scalar output.

        Raises:
            ValueError: If len(inputs) != input_count.
        """

        if len(inputs) != self.input_count:
            raise ValueError("Number of inputs must equal node's input count")
        
        a = self.weights[0] + np.dot(inputs, self.weights[1:])
        return self.phi(a)