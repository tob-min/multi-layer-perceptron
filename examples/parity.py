from multi_layer_perceptron.network import Network
import numpy as np
from multi_layer_perceptron.activations import sigmoid, sigmoid_derivative
import matplotlib.pyplot as plt
from plot_helpers import plot_parity_decision, plot_loss

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

    # Plot both the training points and the current network output surface.
    def display():
        plot_parity_decision(net, coords, classes, grid_x, grid_y, test_coords, output_file=output_file)

    # Show the initial state before training.
    display()

    # Train the network using the higher-level API.
    targets = classes.reshape(-1, 1).astype(float)
    history = net.fit(coords, targets, epochs=epochs, learning_rate=learning_rate)

    # Plot loss vs epoch
    plot_loss(history, output_file=f"{output_file}_loss.png" if output_file else None)

    # Show the final state after training.
    display()

def main():
    parity_problem(output_file="./outputs/graphs/parity_1", learning_rate=0.2, epochs=100)
    parity_problem(output_file="./outputs/graphs/parity_2", learning_rate=0.2, epochs=100, noise = 0.2)
    parity_problem(a_points=np.array([(0, 0), (0., 1.), (1,1)]), 
                   b_points=np.array([(0.5, 0.5), (1, 0), (0.5, 0)]),
                   output_file="./outputs/graphs/parity_3", learning_rate=1, 
                   epochs=100, noise = 0.1, graph_animation=False)


if __name__ == "__main__":
    main()