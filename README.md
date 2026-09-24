# Multi Layer Perceptron

A compact NumPy-based implementation of a multilayer perceptron for learning and experimentation. The project includes a higher-level training API, reusable activation helpers, and example scripts that fit simple regression and parity-style classification tasks.

## Current status

This repository is a small educational implementation rather than a production ML framework. The public API currently centers on the `Network` class in the top-level package:

- `Network(input_count, layer_sizes, activation_functions, activation_derivatives)`
- `Network.forward(inputs)`
- `Network.backprop(eta, output_derivatives)`
- `Network.fit(inputs, targets, epochs=100, learning_rate=0.01, error_derivative=...)`
- `Network.predict(inputs)`
- `Network.evaluate(inputs, targets)`

The package exports the network and supporting layers from `multi_layer_perceptron`.

## Project structure

- `src/multi_layer_perceptron/` — source package
  - `__init__.py` — package exports
  - `network.py` — core network orchestration and training logic
  - `layer.py` — layer implementation and gradient propagation
  - `node.py` — neuron forward pass and local gradient math
  - `output_layer.py` — output layer implementation
  - `output_node.py` — output-node helper for final-layer gradients
  - `activations.py` — activation functions and common derivative helpers
- `examples/` — runnable examples
  - `regression.py` — trains a network to fit a function and saves plots
  - `parity.py` — trains a classifier for a noisy parity-style problem
  - `plot_helpers.py` — helper functions used by the example scripts
- `tests/` — package and API regression tests
- `outputs/graphs/` — generated plots from the example scripts

## Installation

Python 3.9+ is required. Install the package in editable mode from the repository root:

```bash
python -m pip install --upgrade pip
python -m pip install -e .
```

The package dependencies are currently:

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

history = network.fit(x, y, epochs=5, learning_rate=0.01)
predictions = network.predict(x)
mse = network.evaluate(x, y)

print(history)
print(predictions)
print(mse)
```

A more realistic example with a hidden layer is:

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

The higher-level training API is designed around a simple supervised learning flow:

```python
inputs = np.asarray(inputs, dtype=float)
targets = np.asarray(targets, dtype=float)

history = network.fit(inputs, targets, epochs=200, learning_rate=0.05)
predictions = network.predict(inputs)
loss = network.evaluate(inputs, targets)
```

A few important details from the current implementation:

- `fit()` accepts either 1D or 2D arrays and reshapes them automatically when needed.
- Each sample in `inputs` must have length equal to `input_count`.
- Each sample in `targets` must match the output-layer width.
- The default error derivative is `pred - target`, so the loss is effectively the squared error used during training.

## Example scripts

From the repository root, run:

```bash
python examples/regression.py
python examples/parity.py
```

These scripts generate plots to the `outputs/graphs/` directory.

- `examples/regression.py` trains a small network to approximate a function and stores plots such as the regression fit and training loss.
- `examples/parity.py` trains a parity-style classifier and produces a decision boundary plot as well as a loss curve.

## Testing

The project includes a small unittest suite. Run it with:

```bash
pytest
```
or

```bash
python -m unittest discover -s tests
```

## Notes

This package is intentionally minimal and educational. It is useful for understanding forward passes, backpropagation, activation functions, and custom layer construction, but it is not a feature-complete ML toolkit.
