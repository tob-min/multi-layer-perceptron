import os

import matplotlib.pyplot as plt
import numpy as np


def _ensure_parent_dir(path: str | None) -> None:
    if path:
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)


def plot_regression_fit(network, xs, ys, x_test=None, output_file: str | None = None, title: str | None = None):
    xs = np.asarray(xs)
    ys = np.asarray(ys)
    if x_test is None:
        x_test = np.linspace(np.min(xs) - 0.2, np.max(xs) + 0.2, 300)

    predictions = network.predict(x_test).flatten()
    plt.clf()
    plt.scatter(xs, ys, s=8, label="samples")
    plt.plot(x_test, predictions, color="tab:red", linewidth=2, label="network fit")
    plt.title(title or "Network fit")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    if output_file:
        _ensure_parent_dir(output_file)
        plt.savefig(output_file, dpi=200, bbox_inches="tight")
    else:
        plt.show()


def plot_loss(history, output_file: str | None = None):
    history = list(history)
    plt.clf()
    plt.plot(np.arange(1, len(history) + 1), history)
    plt.title("Training Loss vs Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Mean Squared Error")
    if output_file:
        _ensure_parent_dir(output_file)
        plt.savefig(output_file, dpi=200, bbox_inches="tight")
    else:
        plt.show()


def plot_parity_decision(network, coords, classes, grid_x, grid_y, test_coords, output_file: str | None = None):
    preds = network.predict(test_coords).flatten()
    test_classes = (preds > 0.5).astype(int)

    fig = plt.figure(figsize=(12, 5))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122, projection="3d")

    ax1.scatter(coords[:, 0], coords[:, 1], s=18, c=np.where(classes == 1, "#e74c3c", "#3498db"),
                edgecolor="black", linewidth=0.4, label="training points")
    ax1.scatter(test_coords[:, 0], test_coords[:, 1], s=10,
                c=np.where(test_classes == 1, "#f5b7b1", "#b7d9f2"), alpha=0.45, label="decision regions")

    ax1.set_xlim(np.min(grid_x), np.max(grid_x))
    ax1.set_ylim(np.min(grid_y), np.max(grid_y))
    ax1.set_title("Parity classification boundary")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.legend(loc="upper right", frameon=True)

    surface_values = preds.reshape(grid_x.shape)
    ax2.plot_surface(grid_x, grid_y, surface_values, cmap="coolwarm", alpha=0.75, edgecolor="none")
    ax2.set_title("Network output surface")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.set_zlabel("output")
    ax2.set_xlim(np.min(grid_x), np.max(grid_x))
    ax2.set_ylim(np.min(grid_y), np.max(grid_y))
    ax2.view_init(elev=22, azim=-100)

    if output_file:
        _ensure_parent_dir(output_file)
        plt.savefig(output_file, dpi=200, bbox_inches="tight")
    else:
        plt.show()

    plt.close()
