import argparse
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np

from multi_layer_perceptron.activations import *
from multi_layer_perceptron.network import Network
from multi_layer_perceptron.output_node import OutputNode
from plot_helpers import plot_loss, plot_regression_fit


def single_node_test(
    f: Callable[[np.ndarray], np.ndarray] = lambda xs: 2 * xs + 0.1,
    xs: np.ndarray = np.arange(0, 1, 0.1),
    test_xs: np.ndarray = np.arange(-1, 2, 0.02),
    activation: Callable[[float], float] = lambda x: x,
    activation_derivative: Callable[[float], float] = lambda _: 1,
    error_derivative: Callable[[float, float], float] = mean_squared_error_derivative,
    epochs: int = 100000,
    epsilon: float = 0.01,
    output_file: str | None = None,
) -> None:
    """Train a single output node to fit a simple target function."""
    ys = f(xs)
    samples = np.transpose([xs, ys])

    node = OutputNode(activation, activation_derivative)
    weights = np.full(2, 1.0)

    training_indices = np.random.choice(len(xs), epochs)

    for sample in samples[training_indices]:
        x = sample[0]
        y_target = sample[1]
        y_actual = node.calculate(np.array([x]), weights)

        error_gradient = error_derivative(y_actual, y_target)
        delta = node.calc_output_node_gradient(error_gradient)

        weights -= epsilon * np.array([delta, delta * x])

    plt.clf()
    plt.scatter(xs, ys, label="target")
    test_ys = np.array([node.calculate(np.array([x]), weights) for x in test_xs])
    plt.plot(test_xs, test_ys, color="tab:red", linewidth=2, label="single-node fit")
    plt.title("Single-node regression fit")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=200, bbox_inches="tight")
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
    """Train a small network to fit a quartic function and save a clean plot."""
    xs = np.asarray(xs, dtype=float)
    ys = np.random.normal(xs ** 4 - 2 * xs ** 2 + 0.5 * xs + 0.3, noise)

    network = Network(
        input_count=1,
        layer_sizes=network_shape,
        activation_functions=[tanh for _ in range(len(network_shape) - 1)] + [lambda x: x],
        activation_derivatives=[tanh_derivative for _ in range(len(network_shape) - 1)] + [lambda _: 1.0],
    )

    loss = network.fit(xs, ys, epochs, learning_rate)

    if output_file:
        base_path = Path(output_file)
        base_path.parent.mkdir(parents=True, exist_ok=True)
        plot_regression_fit(
            network,
            xs,
            ys,
            x_test=np.linspace(-2.2, 2.2, 300),
            output_file=str(base_path.with_suffix(".png")),
            title="Network fit to a quartic function",
        )
        plot_loss(loss, output_file=str(base_path.parent / f"{base_path.stem}_loss.png"))
    else:
        plot_regression_fit(
            network,
            xs,
            ys,
            x_test=np.linspace(-2.2, 2.2, 300),
            title="Network fit to a quartic function",
        )
        plot_loss(loss)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the multilayer perceptron demo scripts.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--output-dir", type=str, default="outputs/graphs", help="Directory for saved plots.")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs for the quartic demo.")
    parser.add_argument("--learning-rate", type=float, default=0.02, help="Learning rate for the quartic demo.")
    args = parser.parse_args()

    np.random.seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    single_node_test(output_file=str(output_dir / "linear_fit.png"))
    quartic_network_test(
        output_file=str(output_dir / "quartic_network"),
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        network_shape=[10, 1],
    )


if __name__ == "__main__":
    main()