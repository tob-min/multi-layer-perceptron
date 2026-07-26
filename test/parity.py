from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
from network import Network
import numpy as np
from activations import sigmoid, sigmoid_derivative
import matplotlib.pyplot as plt

def parity_problem(a_points: np.ndarray = np.array([(0.,0.), (1.,1.)]),
                b_points: np.ndarray = np.array([(0., 1.), (1., 0.)]),
                noise: float = 0.1,
                network_shape: list[int] = [5,1],
                epochs: int = 10000,
                learning_rate: float = 0.01,
                output_file: str | None = None,
                graph_animation: bool = False) -> None:
    """Train a small network to separate two noisy parity-style classes."""
    # Create the network with sigmoid activations for the hidden layer and a linear output.
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

    # Convert the network output into a hard class decision for plotting.
    def classify(p: float) -> int:
        return 1 if p > 0.5 else 0

    # Plot both the training points and the current network output surface.
    def display():

        fig = plt.figure(figsize=(12, 5))
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122, projection="3d")

        # Left panel: 2D plot of the training samples and predicted regions.
        ax1.scatter(coords[:, 0], coords[:, 1], s=18, c=np.where(classes == 0, "#e74c3c", "#3498db"), edgecolor="black", linewidth=0.4, label="training points")

        test_classes = np.array([classify(net.forward(c)[0]) for c in test_coords])
        ax1.scatter(test_coords[:, 0], test_coords[:, 1], s=10, c=np.where(test_classes == 0, "#f5b7b1", "#b7d9f2"), alpha=0.45, label="decision regions")

        ax1.set_xlim(-1, 2)
        ax1.set_ylim(-1, 2)
        ax1.set_title("Parity classification boundary")
        ax1.set_xlabel("x")
        ax1.set_ylabel("y")
        ax1.legend(loc="upper right", frameon=True)

        # Right panel: 3D surface showing the network output over the input grid.
        surface_values = np.array([net.forward(c)[0] for c in test_coords]).reshape(grid_x.shape)
        ax2.plot_surface(grid_x, grid_y, surface_values, cmap="viridis", alpha=0.85, edgecolor="none")
        ax2.set_title("Network output surface")
        ax2.set_xlabel("x")
        ax2.set_ylabel("y")
        ax2.set_zlabel("output")
        ax2.view_init(elev=22, azim=45)

        if output_file:
            plt.savefig(output_file, dpi=200, bbox_inches="tight")
        else:
            plt.show()

        plt.close()

    # Show the initial state before training.
    display()

    # Train the network by repeatedly updating it from random samples.
    for epoch in range(epochs):
        i = np.random.randint(len(classes))
        output = net.forward(coords[i])
        error_gradient = -2 * (classes[i] - output)
        net.backprop(learning_rate, np.array(error_gradient))

        if graph_animation and 10 * epoch / epochs == (10 * epoch) // epochs:
            display()

    display()

def main():
    parity_problem(output_file="./graphs/parity_1", learning_rate=0.2, epochs=10000)
    parity_problem(output_file="./graphs/parity_2", learning_rate=0.2, epochs=10000, noise = 0.2)
    parity_problem(a_points=np.array([(0, 0), (0., 1.), (1,1)]), 
                   b_points=np.array([(0.5, 0.5), (1, 0), (0.5, 0)]),
                   output_file="./graphs/parity_3", learning_rate=0.2, 
                   epochs=10000, noise = 0.1, graph_animation=False)


if __name__ == "__main__":
    main()