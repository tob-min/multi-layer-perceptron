import unittest

import numpy as np

from multi_layer_perceptron.layer import Layer
from multi_layer_perceptron.output_layer import OutputLayer


class LayerTest(unittest.TestCase):
    def test_layer_calculate_outputs(self) -> None:
        layer = Layer(2, lambda x: x, lambda _: 1.0)
        inputs = np.array([1.0, 2.0], dtype=float)
        weights = np.array([
            [0.5, 0.1],
            [1.0, 2.0],
            [3.0, 4.0],
        ], dtype=float)

        outputs = layer.calculate(inputs, weights)

        self.assertEqual(outputs.shape, (2,))
        self.assertTrue(np.all(np.isfinite(outputs)))

    def test_layer_gradient_calculation(self) -> None:
        layer = Layer(2, lambda x: x, lambda _: 1.0)
        inputs = np.array([1.0, 2.0], dtype=float)
        weights = np.array([
            [0.5, 0.1],
            [1.0, 2.0],
            [3.0, 4.0],
        ], dtype=float)
        layer.calculate(inputs, weights)

        output_weights = np.array([
            [0.0, 0.0],
            [1.0, -1.0],
            [0.25, 0.75],
        ], dtype=float)
        output_dels = np.array([0.5, -0.25], dtype=float)

        gradients = layer.calc_gradients(output_weights, output_dels)

        self.assertEqual(gradients.shape, (2,))
        self.assertTrue(np.all(np.isfinite(gradients)))

    def test_output_layer_gradient_matches_output_nodes(self) -> None:
        output_layer = OutputLayer(2, lambda x: x, lambda _: 1.0)
        output_layer.calculate(np.array([1.0, 2.0]), np.array([
            [0.5, 0.1],
            [1.0, 2.0],
            [3.0, 4.0],
        ], dtype=float))

        gradients = output_layer.calc_output_gradients(np.array([0.5, -0.25], dtype=float))

        self.assertEqual(gradients.shape, (2,))
        self.assertTrue(np.all(np.isfinite(gradients)))

    def test_node_count(self) -> None:
        layer = Layer(3, lambda x: x, lambda _: 1.0)
        self.assertEqual(layer.node_count(), 3)

    def test_multi_node_output_layer(self) -> None:
        output_layer = OutputLayer(3, lambda x: x, lambda _: 1.0)
        outputs = output_layer.calculate(np.array([1.0, 2.0]), np.array([
            [0.5, 0.2, 0.1],
            [1.0, 2.0, 3.0],
            [3.0, 4.0, 5.0],
        ], dtype=float))

        self.assertEqual(outputs.shape, (3,))
        self.assertTrue(np.all(np.isfinite(outputs)))

    def test_invalid_weight_dimension_raises(self) -> None:
        layer = Layer(2, lambda x: x, lambda _: 1.0)
        with self.assertRaisesRegex(ValueError, "Weight array"):
            layer.calculate(np.array([1.0, 2.0]), np.array([1.0, 2.0]))

    def test_layer_rejects_non_matrix_weights(self) -> None:
        layer = Layer(2, lambda x: x, lambda _: 1.0)
        with self.assertRaisesRegex(ValueError, "2D matrix"):
            layer.calculate(np.array([1.0, 2.0]), np.array([1.0, 2.0, 3.0]))

    def test_layer_rejects_wrong_number_of_weight_columns(self) -> None:
        layer = Layer(2, lambda x: x, lambda _: 1.0)
        weights = np.array([
            [0.5, 0.1, 0.2],
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ], dtype=float)

        with self.assertRaisesRegex(ValueError, "one column per node"):
            layer.calculate(np.array([1.0, 2.0]), weights)


if __name__ == "__main__":
    unittest.main()
