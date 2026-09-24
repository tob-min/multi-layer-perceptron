import unittest

import numpy as np

from multi_layer_perceptron import Network


class InputValidationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.network = Network(
            input_count=2,
            layer_sizes=[2, 1],
            activation_functions=[lambda x: x, lambda x: x],
            activation_derivatives=[lambda _: 1.0, lambda _: 1.0],
        )

    def test_init_validates_configuration(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least one layer"):
            Network(2, [], [], [])

        with self.assertRaisesRegex(ValueError, "same size"):
            Network(
                input_count=2,
                layer_sizes=[2, 1],
                activation_functions=[lambda x: x],
                activation_derivatives=[lambda _: 1.0, lambda _: 1.0],
            )

    def test_forward_validates_input_shape(self) -> None:
        with self.assertRaisesRegex(ValueError, "Incorrect number of inputs"):
            self.network.forward(np.array([1.0]))

    def test_backprop_validates_learning_rate_and_output_derivatives(self) -> None:
        self.network.forward(np.array([1.0, 2.0]))

        with self.assertRaisesRegex(ValueError, "Learning rate"):
            self.network.backprop(0.0, np.array([0.0]))

        with self.assertRaisesRegex(ValueError, "output derivatives"):
            self.network.backprop(0.01, np.array([0.0, 0.0]))

    def test_predict_validates_input_samples(self) -> None:
        with self.assertRaisesRegex(ValueError, "Each input sample"):
            self.network.predict(np.array([[1.0, 2.0, 3.0]]))

    def test_evaluate_validates_target_shape(self) -> None:
        inputs = np.array([[0.0, 1.0], [1.0, 2.0]], dtype=float)
        targets = np.array([[0.0, 1.0, 2.0]], dtype=float)

        with self.assertRaisesRegex(ValueError, "same number of samples"):
            self.network.evaluate(inputs, targets)


if __name__ == "__main__":
    unittest.main()
