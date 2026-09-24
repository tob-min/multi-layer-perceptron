"""Activation functions and loss derivatives used by the perceptron layers.

The helpers in this module are intentionally small and explicit so the network
training code remains easy to inspect and modify.
"""

import numpy as np


def mean_squared_error_derivative(y: float, y_target: float) -> float:
    """Return the derivative of the mean squared error with respect to the prediction.

    Args:
        y: Predicted value for the current sample.
        y_target: Ground-truth target value.

    Returns:
        The derivative of the squared error with respect to the prediction.
    """
    # The derivative of (y - target)^2 is 2 * (y - target).
    return 2 * (y - y_target)


def sigmoid(x: float) -> float:
    """Apply the sigmoid activation function to a scalar value.

    Args:
        x: Value passed to the sigmoid function.

    Returns:
        The transformed value in the range (0, 1).
    """
    if x >= 0:
        # Avoid overflow for large positive inputs by using the stable form.
        z = np.exp(-x)
        return 1 / (1 + z)
    else:
        # Use the equivalent formulation for negative inputs.
        z = np.exp(x)
        return z / (1 + z)


def sigmoid_derivative(x: float) -> float:
    """Return the derivative of the sigmoid activation function.

    Args:
        x: Input value to the sigmoid function.

    Returns:
        The derivative of sigmoid(x) with respect to x.
    """
    sigma_x = sigmoid(x)
    return sigma_x * (1 - sigma_x)


def tanh(x: float) -> float:
    """Apply the hyperbolic tangent activation function.

    Args:
        x: Input value.

    Returns:
        The tanh-transformed value.
    """
    return np.tanh(x)


def tanh_derivative(x: float) -> float:
    """Return the derivative of the hyperbolic tangent activation function.

    Args:
        x: Input value.

    Returns:
        The derivative of tanh(x) with respect to x.
    """
    return 1 - np.tanh(x) ** 2