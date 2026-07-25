from node import Node


class Output_Node(Node):
    """Represent a node that emits the network's final prediction."""

    def calc_output_node_gradient(self, error_delta: float) -> float:
        """Return the gradient for an output node from its error delta.

        Args:
            error_delta: Derivative of the loss with respect to this node's output.

        Returns:
            The gradient for the output node's weighted sum.
        """
        return self.activation_prime(self.weighted_sum) * error_delta