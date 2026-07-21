import numpy as np
from numpy import ndarray

from node import Node
from typing import Callable

class Output_Node (Node):
    """Represent an output node in a neural network."""

    output: float
    error_derivative: Callable[[float, float], float]

    def __init__(self, activation_function: Callable[[float], float], 
                 activation_prime: Callable[[float], float], 
                 error_derivative: Callable[[float, float], float]) -> None:
        """Initialize an output node with activation functions and an error derivative.

        Args:
            activation_function: Activation function applied to the node's weighted sum.
            activation_prime: Derivative of activation function.
            error_derivative: Error derivative function used to compute the gradient.
        """
        super().__init__(activation_function, activation_prime)
        self.error_derivative = error_derivative

    def calculate(self, inputs: ndarray, input_weights: ndarray) -> float:
        """Compute the node output for a given input vector and store the result.

        Args:
            inputs: Input array.
            input_weights: Weight array of length len(inputs) + 1.

        Returns:
            Activated scalar output.
        """
        self.output = super().calculate(inputs, input_weights)
        return self.output

    def calc_output_node_gradient(self, y_true: float) -> float:
        """Compute the gradient for the output node given the target value.

        Args:
            y_true: Target output value.

        Returns:
            Calculated gradient.
        """
        output_gradient = self.error_derivative(self.output, y_true)
        return self.activation_prime(self.weighted_sum) * output_gradient