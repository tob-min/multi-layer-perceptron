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
            input_count: Number of input features.
            layer_sizes: Sizes of each layer, including the output layer.
            activation_functions: Activation function for each layer.
            activation_derivatives: Activation-derivative function for each layer.
            error_derivatives: Error-derivative callables validated for the output layer.

        Raises:
            ValueError: If the network is missing layers, if the configuration lists do not match,
                or if there is not one error derivative per output node.
        """
        if not layer_sizes:
            raise ValueError("Network must have at least one layer")

        if not (len(layer_sizes) == len(activation_functions) and len(layer_sizes) == len(activation_derivatives)):
            raise ValueError("layer_sizes, activation_functions, and activation_derivatives must have the same size")

        self.input_count = input_count
        self.weight_matrices = [np.random.uniform(0, 1, size=(input_count+1, layer_sizes[0]))]
        self.layers = []

        for i in range(len(layer_sizes) - 1):
            self.layers.append(Layer(layer_sizes[i], activation_functions[i], activation_derivatives[i]))
            # initialise each weight matrix with random weights
            self.weight_matrices.append(np.random.uniform(0, 1, (layer_sizes[i] + 1, layer_sizes[i + 1])))

        self.output_layer = OutputLayer(layer_sizes[-1], activation_functions[-1], activation_derivatives[-1])
        self.layers.append(self.output_layer)

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """Propagate inputs through the network and return the final outputs."""
        if len(inputs) != self.input_count:
            raise ValueError("Incorrect number of inputs")

        self.inputs = inputs.copy()
        next_inputs = inputs
        for i, layer in enumerate(self.layers):
            next_inputs = layer.calculate(next_inputs, self.weight_matrices[i])
        return next_inputs

    def backprop(self, eta: float, output_derivatives: np.ndarray) -> None:
        """Update all weights using the backpropagation algorithm.

        Args:
            eta: Learning rate.
            output_derivatives: Error derivatives at the network output.
        """

        dels = self.output_layer.calc_output_gradients(output_derivatives)
        for i in range(len(self.layers) - 2, -1, -1):
            dels = self.layers[i].calc_gradients(self.weight_matrices[i+1], dels)

        self.weight_matrices[0] -= eta * (np.append([1],self.inputs)[:, np.newaxis] @ self.layers[0].dels[np.newaxis, :])
        for i in range(1, len(self.weight_matrices)):
            previous_outputs = np.append([1], self.layers[i - 1].outputs)
            current_dels = self.layers[i].dels
            self.weight_matrices[i] -= eta * (previous_outputs[:, np.newaxis] @ current_dels[np.newaxis, :])

    def fit(self, inputs: np.ndarray, targets: np.ndarray, epochs: int = 100, learning_rate: float = 0.01) -> list[float]:
        """Train the network on a simple dataset and return the loss history."""
        inputs = np.asarray(inputs, dtype=float)
        targets = np.asarray(targets, dtype=float)

        if inputs.ndim == 1:
            inputs = inputs.reshape(-1, 1)
        if targets.ndim == 1:
            targets = targets.reshape(-1, 1)

        history: list[float] = []
        for _ in range(epochs):
            for sample_input, target in zip(inputs, targets):
                prediction = self.forward(sample_input)
                error_derivative = np.array([(prediction[0] - target[0])])
                self.backprop(learning_rate, error_derivative)
                history.append(float(np.mean((prediction[0] - target[0]) ** 2)))

        return history

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        """Return predictions for one or more input samples."""
        inputs = np.asarray(inputs, dtype=float)
        if inputs.ndim == 1:
            inputs = inputs.reshape(1, -1)

        predictions = np.array([self.forward(sample_input) for sample_input in inputs])
        return predictions.reshape(-1, 1)

    def evaluate(self, inputs: np.ndarray, targets: np.ndarray) -> float:
        """Return the mean squared error for the provided dataset."""
        predictions = self.predict(inputs)
        targets = np.asarray(targets, dtype=float)
        if targets.ndim == 1:
            targets = targets.reshape(-1, 1)
        return float(np.mean((predictions - targets) ** 2))
