from matplotlib import rcParams
import matplotlib.pyplot as plt
import numpy as np
from node import Node
from output_node import Output_Node


def mean_squared_error_derivative(y: float, y_target: float) -> float:
    """
    Partial derivative of the following error function wrt y
    Error function:
    E = (y-y_target) ** 2
    """
    return 2 * (y - y_target)

def simple_single_node_test(epochs = 100000, epsilon = 0.01, gradient = 2, intercept = 0.1) -> None:
    """Train a single output node to fit a linear function.

    The function uses a simple gradient descent loop over randomly selected
    training samples from a line defined by the supplied slope and intercept.
    The node uses a linear activation and mean squared error, so the update
    rule is derived directly from the error gradient.

    Args:
        epochs: Number of training iterations to run.
        epsilon: Learning rate used for gradient descent.
        gradient: Slope of the target linear function.
        intercept: Intercept of the target linear function.
    """

    xs = np.arange(0, 1, 0.1)
    ys = gradient * xs + intercept
    samples = np.transpose([xs, ys])

    node = Output_Node(lambda x: x, lambda _: 1, mean_squared_error_derivative)
    w = np.full(2, 1.)

    training_indices = np.random.choice(len(xs), epochs)

    for sample in samples[training_indices]:
        x = sample[0]
        y_target = sample[1]
        node.calculate(np.array([x]), w)
        delta = node.calc_output_node_gradient(y_target)

        bias_gradient = delta # * 1
        w1_gradient = delta * x
        w -= epsilon * np.array([bias_gradient, w1_gradient])

    plt.scatter(xs, ys)
    test_xs = np.arange(-1, 2, 0.02)
    test_ys = np.array([node.calculate(np.array([x]), w) for x in test_xs])
    plt.scatter(test_xs, test_ys, s=5)
    plt.show()

def main():
    simple_single_node_test()

if __name__ == "__main__":
    main()