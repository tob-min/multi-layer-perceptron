import unittest

import numpy as np

from multi_layer_perceptron.node import Node
from multi_layer_perceptron.output_node import OutputNode


class NodeTest(unittest.TestCase):
    def test_node_calculate_and_weighted_sum(self) -> None:
        node = Node(lambda x: x, lambda _: 1.0)
        weights = np.array([0.5, 1.0, 2.0], dtype=float)

        value = node.calculate(np.array([3.0, 4.0], dtype=float), weights)

        self.assertAlmostEqual(value, 0.5 + 1.0 * 3.0 + 2.0 * 4.0)
        self.assertAlmostEqual(node.weighted_sum, 0.5 + 1.0 * 3.0 + 2.0 * 4.0)

    def test_node_gradient(self) -> None:
        node = Node(lambda x: x, lambda _: 1.0)
        node.weighted_sum = 3.0

        gradient = node.calc_gradient(np.array([0.2, 0.3]), np.array([1.0, -1.0]))

        self.assertAlmostEqual(gradient, -0.1)

    def test_output_node_gradient(self) -> None:
        node = OutputNode(lambda x: x, lambda _: 1.0)
        node.weighted_sum = 2.5

        gradient = node.calc_output_node_gradient(0.3)

        self.assertAlmostEqual(gradient, 0.3)

    def test_node_rejects_invalid_weight_length(self) -> None:
        node = Node(lambda x: x, lambda _: 1.0)

        with self.assertRaisesRegex(ValueError, "Weight array"):
            node.calculate(np.array([1.0, 2.0]), np.array([1.0]))

    def test_output_node_accepts_scalar_error_delta(self) -> None:
        node = OutputNode(lambda x: x, lambda _: 1.0)
        node.weighted_sum = 4.0

        self.assertAlmostEqual(node.calc_output_node_gradient(0.5), 0.5)

    def test_node_rejects_scalar_input_array(self) -> None:
        node = Node(lambda x: x, lambda _: 1.0)
        with self.assertRaisesRegex(ValueError, "Weight array"):
            node.calculate(np.array(2.0), np.array([0.5, 1.0, 2.0]))


if __name__ == "__main__":
    unittest.main()
