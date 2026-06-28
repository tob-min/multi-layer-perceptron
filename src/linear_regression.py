from matplotlib import rcParams
import matplotlib.pyplot as plt
import numpy as np
from node import Perceptron_Node as PN

"""
Error function:
E = sum((y-z) ** 2)
"""

def d0(ys: np.ndarray, zs: np.ndarray) -> float:
    return -2 * np.sum(ys - zs)

def d1(xs: np.ndarray, ys: np.ndarray, zs: np.ndarray) -> float:
    return -2 * np.sum(xs * (ys-zs))

def gradient_descent (node: PN, xs: np.ndarray, ys: np.ndarray, eta: float = 1e-2, iters: int = 10000) -> None:
    for i in range(iters):
        zs = np.array([node.calculate(np.array([xs[i]])) for i in range(len(xs))])
        d = np.array([d0(ys, zs), d1(xs, ys, zs)])
        node.set_weights(node.weights - eta * d)

def linear_regression(xs: np.ndarray, ys: np.ndarray) -> None:

    n = PN(1, np.array([0, 0]), lambda x: x)

    gradient_descent (n, xs, ys)

    z = np.array([n.calculate(np.array([x])) for x in xs])
    print("Weights: "+ str(n.weights))
    print("Derivatives: " + str([d0(ys, z), d1(xs, ys, z)]))

    plt.scatter(xs, ys)

    x2 = np.arange(min(xs), max(xs), (max(xs)-min(xs)) / 100)
    zs = np.array([n.calculate(np.array([x])) for x in x2])
    plt.scatter(x2, zs, s=rcParams['lines.markersize'] ** 2 / 5)
    plt.show()

def main():
    xs = np.arange(0, 1, 0.1)
    ys = 2* xs + 0.2

    linear_regression(xs, ys)

if __name__ == "__main__":
    main()