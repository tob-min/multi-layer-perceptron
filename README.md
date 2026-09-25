# Multi-Layer Perceptron

A compact NumPy-based implementation of a multilayer perceptron designed for learning, experimentation, and demos. The project focuses on the fundamentals of neural-network training: forward propagation, backpropagation, activation functions, and gradient-based optimization.

## Overview

This repository is intentionally small and educational. It is not a production ML framework, but it does provide a working end-to-end implementation of a feedforward network with a clean high-level API.

The package centers on the `Network` class and exposes a simple training workflow for supervised learning tasks:

- `Network(input_count, layer_sizes, activation_functions, activation_derivatives)`
- `Network.forward(inputs)`
- `Network.backprop(eta, output_derivatives)`
- `Network.fit(inputs, targets, epochs=100, learning_rate=0.01, error_derivative=...)`
- `Network.predict(inputs)`
- `Network.evaluate(inputs, targets)`

## Why this project exists

This project was built to explore how neural networks work at a low level:

- weights and biases are updated manually
- activations are defined explicitly
- gradients are propagated through each layer
- the training loop is designed to be readable rather than abstracted away

That makes it useful for understanding the mechanics behind modern ML libraries without depending on PyTorch or TensorFlow.

## Repository structure

- `src/multi_layer_perceptron/` — package source code
  - `network.py` — training loop, forward pass, and prediction utilities
  - `layer.py` — hidden-layer calculation and local gradient logic
  - `node.py` — single-neuron activation and weight math
  - `output_layer.py` — final-layer gradient handling
  - `output_node.py` — scalar output-node utility used in didactic examples
  - `activations.py` — sigmoid and tanh helper functions and derivatives
- `examples/` — runnable example scripts
  - `regression.py` — fits a function and saves plotting output
  - `parity.py` — trains a noisy parity-style classifier
  - `plot_helpers.py` — plotting utilities for the examples
- `tests/` — regression tests covering validation, training, and API behavior
- `outputs/graphs/` — generated plots from example runs

## Installation

Python 3.9+ is required. Install the project in editable mode from the repository root:

```bash
python -m pip install --upgrade pip
python -m pip install -e .
```

Dependencies:

- `numpy`
- `matplotlib`

## Quick start

```python
import numpy as np
from multi_layer_perceptron import Network
from multi_layer_perceptron.activations import sigmoid, sigmoid_derivative

x = np.array([[0.0], [1.0], [2.0]], dtype=float)
y = np.array([[0.0], [1.0], [2.0]], dtype=float)

network = Network(
    input_count=1,
    layer_sizes=[1],
    activation_functions=[lambda z: z],
    activation_derivatives=[lambda _: 1.0],
)

history = network.fit(x, y, epochs=20, learning_rate=0.01)
predictions = network.predict(x)
mse = network.evaluate(x, y)

print(history)
print(predictions)
print(mse)
```

A simple hidden-layer example:

```python
from multi_layer_perceptron import Network
from multi_layer_perceptron.activations import sigmoid, sigmoid_derivative

network = Network(
    input_count=2,
    layer_sizes=[4, 1],
    activation_functions=[sigmoid, sigmoid],
    activation_derivatives=[sigmoid_derivative, lambda _: 1.0],
)

features = [[0.2, -0.7], [1.0, 0.5]]
print(network.forward(features[0]))
```

## Training workflow

The project supports a straightforward supervised-learning pattern:

```python
inputs = np.asarray(inputs, dtype=float)
targets = np.asarray(targets, dtype=float)

history = network.fit(inputs, targets, epochs=200, learning_rate=0.05)
predictions = network.predict(inputs)
loss = network.evaluate(inputs, targets)
```

A few implementation details are worth noting:

- `fit()` accepts both 1D and 2D inputs and reshapes automatically when needed.
- Each input sample must match the configured `input_count`.
- Each target sample must match the output-layer width.
- By default, the loss derivative is defined as `prediction - target`.

## Example scripts

From the repository root, run:

```bash
python examples/regression.py
python examples/parity.py
```

The example scripts generate plots in `outputs/graphs/`:

- `regression.py` trains a network to approximate a target function.
- `parity.py` trains a noisy parity-style classifier and visualizes the decision boundary.

## Testing

The project includes a small test suite built with `unittest` and `pytest`-compatible conventions.

Run it with:

```bash
python -m pytest -q
```

or:

```bash
python -m unittest discover -s tests
```

## Current scope and limitations

This implementation is intentionally minimal and is best understood as a teaching project. It is useful for learning:

- gradient descent
- backpropagation
- activation functions
- layer-to-layer weight updates

It is not a general-purpose ML framework and does not include advanced tooling such as:

- distributed training
- batch optimization libraries
- GPU acceleration
- production model serialization
- a full training pipeline for large datasets

## License

This project is released under the MIT License.
