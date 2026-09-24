"""Public package interface for the multilayer perceptron implementation.

This module exposes the core classes used to construct, train, and evaluate
small neural networks built from NumPy arrays and simple activation helpers.
"""

from .layer import Layer
from .network import Network
from .node import Node
from .output_layer import OutputLayer
from .output_node import OutputNode
from . import activations

__all__ = [
    "Layer",
    "Network",
    "Node",
    "OutputLayer",
    "OutputNode",
    "activations",
]
