"""Output-node helper for single-output training examples.

This small helper mirrors the final-layer gradient logic used by the training
network, but in a more direct scalar form.
"""

from .node import Node


class OutputNode(Node):
    """Represent a node that emits the network's final prediction."""

    def calc_output_node_gradient(self, error_delta: float) -> float:
        """Return the gradient for an output node from its error delta.

        Args:
            error_delta: Derivative of the loss with respect to this node's output.

        Returns:
            The gradient for the output node's weighted sum.
        """
        # Apply the chain rule to convert the loss derivative into a weighted-sum gradient.
        return self.activation_prime(self.weighted_sum) * error_delta