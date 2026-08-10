from typing import Callable

import matplotlib.pyplot as plt
import numpy as np

from src.network import Network
from src.output_node import OutputNode
from test.activations import *

def single_node_test(f: Callable[[np.ndarray], np.ndarray] = lambda xs: 2*xs + 0.1, 
                    xs : np.ndarray = np.arange(0, 1, 0.1),
                    test_xs: np.ndarray = np.arange(-1, 2, 0.02),
                    activation: Callable[[float], float] = lambda x: x,
                    activation_derivative: Callable[[float], float] = lambda _: 1,
                    error_derivative: Callable[[float, float], float] = mean_squared_error_derivative,
                    epochs = 100000, epsilon = 0.01,
                    output_file: str | None = None) -> None:
    """Train a single output node to fit a target function.

    The function uses a simple gradient descent loop over randomly selected
    training samples generated from the supplied target function. The node can
    use a custom activation function and derivative, allowing the same training
    routine to be reused for different nonlinear behaviours. The learning rate
    and number of iterations are configurable through the arguments.

    Args:
        f: Function that produces target values for the input array.
        xs: Training inputs used to generate the target samples.
        test_xs: Inputs used to evaluate the trained node after fitting.
        activation: Activation function applied by the node.
        activation_derivative: Derivative of the activation function.
        error_derivative: Derivative of the error function used for training.
        epochs: Number of training iterations to run.
        epsilon: Learning rate used for gradient descent.
        output_file: Optional file path for saving the generated plot.
    """

    ys = f(xs)
    samples = np.transpose([xs, ys])

    node = OutputNode(activation, activation_derivative)
    w = np.full(2, 1.)

    training_indices = np.random.choice(len(xs), epochs)

    for sample in samples[training_indices]:
        x = sample[0]
        y_target = sample[1]
        y_actual = node.calculate(np.array([x]), w)

        error_gradient = error_derivative(y_actual, y_target)
        delta = node.calc_output_node_gradient(error_gradient)

        bias_gradient = delta # * 1
        w1_gradient = delta * x
        w -= epsilon * np.array([bias_gradient, w1_gradient])

    plt.clf()
    plt.scatter(xs, ys)
    test_ys = np.array([node.calculate(np.array([x]), w) for x in test_xs])
    plt.scatter(test_xs, test_ys, s=5)
    if output_file:
        plt.savefig(output_file)
    else:
        plt.show()

def quartic_network_test(
    xs: np.ndarray = np.linspace(-2, 2, 200),
    epochs: int = 2000,
    learning_rate: float = 0.01,
    output_file: str | None = None,
    noise: float = 0.1,
    network_shape: list[int] = [8, 1],
) -> None:
    """Train a small network to fit a quartic function and plot the result."""
    xs = np.asarray(xs, dtype=float)
    ys = np.random.normal(xs ** 4 - 2 * xs ** 2 + 0.5 * xs + 0.3, noise)

    def display() -> None:

        x_test = np.linspace(-2.2, 2.2, 300)
        predictions = np.array([network.forward(np.array([x])) for x in x_test])
        plt.clf()
        plt.scatter(xs, ys, s=8, label="quartic samples")
        plt.plot(x_test, predictions, color="tab:red", linewidth=2, label="network fit")
        plt.title("Network fit to a quartic function")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend()
    
        if output_file:
            plt.savefig(output_file)
        else:
            plt.show()

    network = Network(
        input_count=1,
        layer_sizes=network_shape,
        activation_functions=[tanh for _ in range(len(network_shape) - 1)] + [lambda x: x],
        activation_derivatives=[tanh_derivative for _ in range(len(network_shape) - 1)] + [lambda _: 1.],
    )

    for epoch in range(epochs):
        i = np.random.choice(len(xs))
        predictions = network.forward(xs[i:i+1])
        error_derivative = np.array([(predictions[0] - ys[i])])
        network.backprop(learning_rate, error_derivative)

        if 10 * epoch / epochs == (10 * epoch) // epochs:
            display()

    x_test = np.linspace(-2.2, 2.2, 300)
    predictions = np.array([network.forward(np.array([x])) for x in x_test])

    display()

def main():
    single_node_test(output_file="./graphs/linear")
    quartic_network_test(output_file="./graphs/quartic_network", 
                         epochs=10000, learning_rate=0.01, network_shape=[10, 1])


if __name__ == "__main__":
    main()