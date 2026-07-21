import numpy as np
from typing import Callable

class Node:

    activation: Callable
    activation_prime: Callable
    weighted_sum: float

    def __init__(self, activation_function: Callable[[float], float], activation_prime: Callable[[float], float]) -> None:
        """Initialize a node with an activation function.

        Args:
            activation_function: Activation function applied to the node's weighted sum.
            activation_prime: Derivative of activation function

        Raises:
            ValueError: If len(weights) != input_count + 1.
        """
        
        self.activation = activation_function
        self.activation_prime = activation_prime

    def calculate(self, inputs: np.ndarray, input_weights: np.ndarray) -> float:
        """Compute the node output for a given input vector and store weighted sum.

        Args:
            inputs: Input array.
            input_weights: Weight array of length len(inputs) + 1.

        Returns:
            Activated scalar output.

        Raises:
            ValueError: If len(inputs) != input_count.
        """

        if len(input_weights) != len(inputs) + 1:
            raise ValueError("Weight array must be of length one greater than input array")
        
        self.weighted_sum = input_weights[0] + np.dot(inputs, input_weights[1:])
        return self.activation(self.weighted_sum)

    def calc_gradient(self, output_weights: np.ndarray, output_dels: np.ndarray) -> float:
        """Compute del E/del a for E the overall error and a the weighted sum in this node
        
        Args: 
            output_dels: Input array of del E/ del ak for the weighed sum ak of each output node k.
            output_weights: Output weight array of same dimensions as output_dels

        Returns:
            Calculated gradient
        """
        return self.activation_prime(self.weighted_sum) * np.dot(output_weights, output_dels)