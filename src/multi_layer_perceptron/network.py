"""Core multilayer perceptron implementation.

This module contains the primary training and inference logic for the project,
including forward propagation, backpropagation, and the high-level fit/predict/evaluate API.
"""

from .layer import Layer
from .output_layer import OutputLayer
from typing import Callable
import numpy as np


class Network:
    """Represent a multilayer perceptron composed of layers and weight matrices."""

    layers: list[Layer]
    output_layer: OutputLayer
    weight_matrices: list[np.ndarray] # list of matrices of weights between each layer
    input_count: int
    inputs: np.ndarray


    def __init__(self, input_count: int, layer_sizes: list[int], activation_functions: list[Callable[[float], float]],
                 activation_derivatives: list[Callable[[float], float]]) -> None:
        """Initialize the network and its weight matrices.

        Args:
            input_count: Number of input features for each training sample.
            layer_sizes: Sizes of each layer, including the output layer.
            activation_functions: Activation function for each layer.
            activation_derivatives: Activation-derivative function for each layer.

        Raises:
            ValueError: If the network is missing layers, if the configuration lists do not match,
                or if any layer definition is invalid.
        """
        if input_count <= 0:
            raise ValueError("input_count must be a positive integer")

        if layer_sizes == []:
            raise ValueError("Network must have at least one layer")

        if any(size <= 0 for size in layer_sizes):
            raise ValueError("layer_sizes must contain positive integers")

        if not (len(layer_sizes) == len(activation_functions) and len(layer_sizes) == len(activation_derivatives)):
            raise ValueError("layer_sizes, activation_functions, and activation_derivatives must have the same size")

        if not all(callable(fn) for fn in activation_functions):
            raise ValueError("activation_functions must contain callables for every layer")

        if not all(callable(fn) for fn in activation_derivatives):
            raise ValueError("activation_derivatives must contain callables for every layer")

        self.input_count = input_count
        self.weight_matrices = [np.random.uniform(0, 1, size=(input_count + 1, layer_sizes[0]))]
        self.layers = []

        for i in range(len(layer_sizes) - 1):
            self.layers.append(Layer(layer_sizes[i], activation_functions[i], activation_derivatives[i]))
            # initialise each weight matrix with random weights
            self.weight_matrices.append(np.random.uniform(0, 1, (layer_sizes[i] + 1, layer_sizes[i + 1])))

        self.output_layer = OutputLayer(layer_sizes[-1], activation_functions[-1], activation_derivatives[-1])
        self.layers.append(self.output_layer)


    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """Propagate a single sample through the network and return the final outputs.

        Args:
            inputs: One input sample of length `input_count`, or a batch of samples shaped
                `(n_samples, input_count)`.

        Returns:
            The network output for the supplied input(s).

        Raises:
            ValueError: If the input shape does not match the network input count.
        """
        inputs = np.asarray(inputs, dtype=float)

        if inputs.ndim != 1 or len(inputs) != self.input_count:
            raise ValueError(f"Incorrect number of inputs: must be a 1D sample of length {self.input_count}")
        
        self.inputs = inputs.copy()
        next_inputs = inputs

        for i, layer in enumerate(self.layers):
            next_inputs = layer.calculate(next_inputs, self.weight_matrices[i])
            
        return next_inputs


    def backprop(self, eta: float, output_derivatives: np.ndarray) -> None:
        """Update all weights using the backpropagation algorithm.

        Args:
            eta: Learning rate used to scale the weight update.
            output_derivatives: Error derivatives at the network output, shaped to match the
                number of output nodes.

        Raises:
            ValueError: If `eta` is not positive or the output derivative shape does not match the
                output layer.
        """
        if eta <= 0:
            raise ValueError("Learning rate must be a positive number")

        output_derivatives = np.asarray(output_derivatives, dtype=float)
        
        if output_derivatives.ndim == 0:
            output_derivatives = output_derivatives.reshape(1)

        if output_derivatives.shape != (len(self.output_layer.nodes),):
            raise ValueError("output derivatives must have length equal to the output layer size")

        if not hasattr(self, "inputs"):
            raise ValueError("Network inputs are not set; call forward() before backprop()")

        dels = self.output_layer.calc_output_gradients(output_derivatives)
        for i in range(len(self.layers) - 2, -1, -1):
            dels = self.layers[i].calc_gradients(self.weight_matrices[i + 1], dels)

        self.weight_matrices[0] -= eta * (np.append([1], self.inputs)[:, np.newaxis] @ self.layers[0].dels[np.newaxis, :])
        for i in range(1, len(self.weight_matrices)):
            previous_outputs = np.append([1], self.layers[i - 1].outputs)
            current_dels = self.layers[i].dels
            self.weight_matrices[i] -= eta * (previous_outputs[:, np.newaxis] @ current_dels[np.newaxis, :])


    def fit(self, inputs: np.ndarray, targets: np.ndarray, epochs: int = 100, learning_rate: float = 0.01,
            error_derivative: Callable[[np.ndarray, np.ndarray], np.ndarray] = (lambda pred, target: pred - target)
            ) -> list[float]:
        """Train the network on a simple dataset and return the loss history.

        Args:
            inputs: Array-like dataset of float samples, each of length `input_count`.
            targets: Array-like target values, each matching the output layer width.
            epochs: Number of training epochs to run.
            learning_rate: Step size used by gradient descent.
            error_derivative: Function that returns the derivative of the loss with respect to
                each output prediction. It receives the predicted value and target value for a
                given sample and should return the local derivative.

        Returns:
            A list of mean error values recorded at each epoch.

        Raises:
            ValueError: If the inputs and targets are incompatible with the network shape or if the
                training configuration is invalid.
        """
        if epochs <= 0:
            raise ValueError("epochs must be a positive integer")

        if learning_rate <= 0:
            raise ValueError("learning_rate must be a positive number")

        inputs = np.asarray(inputs, dtype=float)
        targets = np.asarray(targets, dtype=float)
        
        if inputs.size == 0 or targets.size == 0:
            raise ValueError("There must be at least one sample")

        if inputs.ndim == 1:
            inputs = inputs.reshape(-1, 1)
        if targets.ndim == 1:
            targets = targets.reshape(-1, 1)

        # Validation: inputs and targets must have the same number of samples
        if inputs.shape[0] != targets.shape[0]:
            raise ValueError("Inputs and targets must have the same number of samples")

        # After reshaping, each input sample must have length `input_count`
        if inputs.shape[1] != self.input_count:
            raise ValueError(f"Each input sample must have length {self.input_count}")

        # Each target sample must have the same length as the output layer
        expected_output_len = len(self.output_layer.nodes)
        if targets.shape[1] != expected_output_len:
            raise ValueError(f"Each target sample must have length {expected_output_len}")

        history: list[float] = []
        for _ in range(epochs):
            # deliver input samples in a random order
            order = np.random.permutation(len(inputs))
            error = 0
            
            for i in order:
                prediction = self.forward(inputs[i])
                self.backprop(learning_rate, error_derivative(prediction, targets[i]))
                # uses mean squared error TODO: allow arbitrary error function
                error += np.sum((prediction - targets[i]) ** 2)
                
            # return mean error
            history.append(error / len(inputs))

        return history


    def predict(self, inputs: np.ndarray) -> np.ndarray:
        """Return predictions for one or more input samples.

        Args:
            inputs: A single input sample or a batch of samples with shape `(n_samples, input_count)`
                    (or shape (n_samples,) if the input_count is 1)

        Returns:
            A 2D NumPy array containing the predicted outputs for each input sample.

        Raises:
            ValueError: If the supplied samples do not match the configured input count.
        """
        inputs = np.asarray(inputs, dtype=float)
        
        if inputs.size == 0:
            raise ValueError("There must be at least one sample")

        if inputs.ndim == 1:
            # if there is only one input, interpret a 1D array as a batch of samples
            if self.input_count == 1:
                inputs = inputs.reshape(-1, 1)
            # otherwise interpret it as a single sample
            else:
                inputs = inputs.reshape(1, -1)
                
        elif inputs.ndim != 2:
            raise ValueError("Inputs must be a 1D sample or a 2D array of samples")

        if inputs.shape[1] != self.input_count:
            raise ValueError(f"Each input sample must have length {self.input_count}")

        # pass each sample through the network
        predictions = np.array([self.forward(sample_input) for sample_input in inputs])
        return predictions.reshape(-1, 1) if predictions.ndim == 1 else predictions


    def evaluate(self, inputs: np.ndarray, targets: np.ndarray) -> float:
        """Return the mean squared error for the provided dataset.

        Args:
            inputs: Array-like dataset of input samples.
            targets: Array-like target values aligned with the inputs.

        Returns:
            The mean squared error between predictions and targets.

        Raises:
            ValueError: If the inputs and targets are incompatible with each other or with the
                network configuration.
        """
            
        inputs = np.asarray(inputs, dtype=float)
        targets = np.asarray(targets, dtype=float)
        
        if inputs.size == 0 or targets.size == 0:
            raise ValueError("There must be at least one sample")

        if inputs.ndim < 2:
            inputs = inputs.reshape(-1, 1)
        if targets.ndim < 2:
            targets = targets.reshape(-1, 1)

        if inputs.shape[0] != targets.shape[0]:
            raise ValueError("Inputs and targets must have the same number of samples")
        
        if (targets.shape[1] != self.output_layer.node_count()):
            raise ValueError(f"Each target sample must have length {self.output_layer.node_count()}")

        predictions = self.predict(inputs)

        return float(np.mean((predictions - targets) ** 2))
