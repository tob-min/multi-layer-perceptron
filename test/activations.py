import numpy as np

def mean_squared_error_derivative(y: float, y_target: float) -> float:
    """Return the derivative of the mean squared error with respect to the prediction."""
    return 2 * (y - y_target)

def sigmoid(x: float) -> float:
    """Apply the sigmoid activation function to a scalar value."""
    if x >= 0:
        z = np.exp(-x)
        return 1 / (1 + z)
    else:
        z = np.exp(x)
        return z / (1 + z)

def sigmoid_derivative(x: float) -> float:
    """Return the derivative of the sigmoid activation function."""
    sigma_x = sigmoid(x)
    return sigma_x * (1 - sigma_x)

def tanh(x: float) -> float:
    return np.tanh(x)

def tanh_derivative(x: float) -> float:
    return 1 - np.tanh(x) ** 2