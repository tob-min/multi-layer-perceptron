import argparse
from pathlib import Path

import numpy as np

from multi_layer_perceptron.activations import sigmoid, sigmoid_derivative
from multi_layer_perceptron.network import Network
from plot_helpers import plot_loss, plot_parity_decision


def parity_problem(
    a_points: np.ndarray = np.array([(0.0, 0.0), (1.0, 1.0)]),
    b_points: np.ndarray = np.array([(0.0, 1.0), (1.0, 0.0)]),
    noise: float = 0.1,
    network_shape: list[int] = [5, 1],
    epochs: int = 10000,
    learning_rate: float = 0.01,
    output_file: str | None = None,
) -> None:
    """Train a small network to separate two noisy parity-style classes."""
    net = Network(
        input_count=2,
        layer_sizes=network_shape,
        activation_functions=[sigmoid for _ in range(len(network_shape))],
        activation_derivatives=[sigmoid_derivative for _ in range(len(network_shape))],
    )

    # Sample a set of noisy points belonging to either class 0 or class 1.
    classes = np.random.choice([0, 1], size=100)
    coords = np.empty((len(classes), 2), dtype=float)
    for i, class_label in enumerate(classes):
        base_points = a_points if class_label == 0 else b_points
        base_point = base_points[np.random.randint(len(base_points))]
        coords[i] = np.random.normal(loc=base_point, scale=noise, size=2)

    # Build a dense grid for visualizing the learned decision boundary.
    x_vals = np.linspace(-1, 2, 100)
    y_vals = np.linspace(-1, 2, 100)
    grid_x, grid_y = np.meshgrid(x_vals, y_vals)
    test_coords = np.column_stack((grid_x.ravel(), grid_y.ravel()))
    
    targets = classes.reshape(-1, 1).astype(float)
    history = net.fit(coords, targets, epochs=epochs, learning_rate=learning_rate)

    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plot_loss(history, output_file=str(output_path.parent / f"{output_path.stem}_loss.png"))
        plot_parity_decision(net, coords, classes, grid_x, grid_y, test_coords, output_file=str(output_path.with_suffix(".png")))
    else:
        plot_loss(history)
        plot_parity_decision(net, coords, classes, grid_x, grid_y, test_coords)

def main() -> None:
    parser = argparse.ArgumentParser(description="Train parity-style classification demos.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--output-dir", type=str, default="outputs/graphs", help="Directory for saved plots.")
    parser.add_argument("--epochs", type=int, default=150, help="Training epochs.")
    parser.add_argument("--learning-rate", type=float, default=0.2, help="Learning rate.")
    parser.add_argument("--noise", type=float, default=0.1, help="Standard deviation for noisy samples.")
    args = parser.parse_args()

    np.random.seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    parity_problem(output_file=str(output_dir / "parity_1"), learning_rate=args.learning_rate, epochs=args.epochs, noise=args.noise)
    parity_problem(output_file=str(output_dir / "parity_2"), learning_rate=args.learning_rate, epochs=args.epochs, noise=args.noise + 0.1)


if __name__ == "__main__":
    main()