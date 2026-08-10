# Multi Layer Perceptron

A small educational implementation of a multilayer perceptron (MLP) in Python using NumPy. The project is designed to be easy to read and to demonstrate the mechanics of forward propagation and backpropagation.

## Project structure

- src/multi_layer_perceptron/ — core package implementation
  - network.py — network orchestration and training updates
  - layer.py — hidden-layer logic
  - node.py — individual neuron computations
  - output_layer.py — output-layer-specific gradient handling
  - output_node.py — output-node helper for training
  - activations.py — activation and loss helper functions
- examples/ — runnable example scripts
  - regression.py — fits a simple function and plots the result
  - parity.py — demonstrates a parity-style classification problem
- tests/ — regression and package-import tests

## Installation

This project uses Python 3.9+ and depends on NumPy and Matplotlib.

```bash
python -m pip install -U pip
python -m pip install numpy matplotlib
```

## Running the examples

From the repository root, run:

```bash
python examples/regression.py
python examples/parity.py
```

These scripts will generate plots under the outputs/graphs/ directory.

## Example usage from Python

```python
from multi_layer_perceptron.network import Network
from multi_layer_perceptron.activations import sigmoid, sigmoid_derivative

network = Network(
    input_count=2,
    layer_sizes=[4, 1],
    activation_functions=[sigmoid, sigmoid],
    activation_derivatives=[sigmoid_derivative, lambda _: 1.0],
)

outputs = network.forward([0.2, -0.7])
print(outputs)
```

## Notes

This repository is intentionally simple and educational. It is a good starting point for understanding how MLPs work under the hood, but it does not yet include the full suite of production-oriented features that larger libraries provide.
