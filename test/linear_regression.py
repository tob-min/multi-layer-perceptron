import matplotlib.pyplot as plt
import numpy as np
from node import Node
from output_node import Output_Node
from typing import Callable

def mean_squared_error_derivative(y: float, y_target: float) -> float:
    """Return the derivative of the mean squared error with respect to the prediction."""
    return 2 * (y - y_target)

def sigmoid(x: float) -> float:
    """Apply the sigmoid activation function to a scalar value."""
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x: float) -> float:
    """Return the derivative of the sigmoid activation function."""
    sigma_x = sigmoid(x)
    return sigma_x * (1 - sigma_x)

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

    node = Output_Node(activation, activation_derivative)
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

def main():
    single_node_test(output_file="./graphs/linear.jpg")

if __name__ == "__main__":
    main()