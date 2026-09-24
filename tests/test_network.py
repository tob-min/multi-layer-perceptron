import unittest

import numpy as np

from multi_layer_perceptron import Network


class NetworkTest(unittest.TestCase):
            
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

        with self.assertRaisesRegex(ValueError, "positive integer"):
            Network(0, [1], [lambda x: x], [lambda _: 1.0])

    def test_forward_and_predict_shapes(self) -> None:
        network = Network(
            input_count=2,
            layer_sizes=[2, 1],
            activation_functions=[lambda x: x, lambda x: x],
            activation_derivatives=[lambda _: 1.0, lambda _: 1.0],
        )

        forward_output = network.forward(np.array([1.0, 2.0], dtype=float))
        self.assertEqual(forward_output.shape, (1,))

        predictions = network.predict(np.array([[1.0, 2.0], [0.0, 1.0]], dtype=float))
        self.assertEqual(predictions.shape, (2, 1))

        with self.assertRaisesRegex(ValueError, "Incorrect number of inputs"):
            network.forward(np.array([1.0], dtype=float))

        with self.assertRaisesRegex(ValueError, "Each input sample"):
            network.predict(np.array([[1.0]], dtype=float))

    def test_fit_predict_and_evaluate_work(self) -> None:
        network = Network(
            input_count=1,
            layer_sizes=[1],
            activation_functions=[lambda x: x],
            activation_derivatives=[lambda _: 1.0],
        )

        inputs = np.array([[0.0], [1.0], [2.0]], dtype=float)
        targets = np.array([[0.0], [1.0], [2.0]], dtype=float)

        history = network.fit(inputs, targets, epochs=5, learning_rate=0.01)
        predictions = network.predict(inputs)
        mse = network.evaluate(inputs, targets)

        self.assertEqual(len(history), 5)
        self.assertEqual(predictions.shape, targets.shape)
        self.assertGreaterEqual(mse, 0.0)

    def test_fit_validates_input_target_shapes(self) -> None:
        network = Network(
            input_count=2,
            layer_sizes=[2, 2],
            activation_functions=[lambda x: x, lambda x: x],
            activation_derivatives=[lambda _: 1.0, lambda _: 1.0],
        )

        inputs = np.array([[0.0, 1.0], [1.0, 2.0]], dtype=float)

        with self.assertRaisesRegex(ValueError, "same number of samples"):
            network.fit(inputs, np.array([[0.0]], dtype=float), epochs=1)

        with self.assertRaisesRegex(ValueError, "Each target sample"):
            network.fit(inputs, np.array([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], dtype=float), epochs=1)

    def test_backprop_rejects_invalid_learning_rate_and_output_shape(self) -> None:
        network = Network(
            input_count=2,
            layer_sizes=[1, 1],
            activation_functions=[lambda x: x, lambda x: x],
            activation_derivatives=[lambda _: 1.0, lambda _: 1.0],
        )
        network.forward(np.array([1.0, 2.0], dtype=float))

        with self.assertRaisesRegex(ValueError, "Learning rate"):
            network.backprop(0.0, np.array([0.5]))

        with self.assertRaisesRegex(ValueError, "output derivatives"):
            network.backprop(0.1, np.array([0.5, 0.5]))

    def test_multi_output_network(self) -> None:
        network = Network(
            input_count=2,
            layer_sizes=[3, 2],
            activation_functions=[lambda x: x, lambda x: x],
            activation_derivatives=[lambda _: 1.0, lambda _: 1.0],
        )

        inputs = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=float)
        targets = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=float)

        history = network.fit(inputs, targets, epochs=3, learning_rate=0.01)
        predictions = network.predict(inputs)
        mse = network.evaluate(inputs, targets)

        self.assertEqual(predictions.shape, (2, 2))
        self.assertEqual(len(history), 3)
        self.assertGreaterEqual(mse, 0.0)

    def test_empty_or_scalar_inputs_are_rejected(self) -> None:
        network = Network(
            input_count=2,
            layer_sizes=[1],
            activation_functions=[lambda x: x],
            activation_derivatives=[lambda _: 1.0],
        )

        with self.assertRaisesRegex(ValueError, "Incorrect number of inputs"):
            network.forward(np.array([], dtype=float))

        with self.assertRaisesRegex(ValueError, "Inputs must be a 1D sample"):
            network.predict(np.array(1.0, dtype=float))

    def test_evaluate_error_mentions_numeric_target_length(self) -> None:
        network = Network(
            input_count=2,
            layer_sizes=[1],
            activation_functions=[lambda x: x],
            activation_derivatives=[lambda _: 1.0],
        )

        inputs = np.array([
            [0.0, 1.0],
            [1.0, 2.0],
        ], dtype=float)
        targets = np.array([
            [0.0, 1.0],
            [1.0, 2.0],
        ], dtype=float)

        with self.assertRaisesRegex(ValueError, "length 1"):
            network.evaluate(inputs, targets)

    def test_fit_should_reject_empty_training_set_cleanly(self) -> None:
        network = Network(
            input_count=2,
            layer_sizes=[1],
            activation_functions=[lambda x: x],
            activation_derivatives=[lambda _: 1.0],
        )

        inputs = np.empty((0, 2), dtype=float)
        targets = np.empty((0, 1), dtype=float)

        with self.assertRaisesRegex(ValueError, "at least one sample"):
            network.fit(inputs, targets, epochs=1)

    def test_fit_should_reject_non_numeric_activations(self) -> None:
        with self.assertRaisesRegex(ValueError, "callables"):
            Network(
                input_count=2,
                layer_sizes=[2, 1],
                activation_functions=[lambda x: x, "not callable"], # type: ignore[arg-type]
                activation_derivatives=[lambda _: 1.0, lambda _: 1.0],
            )

    def test_predict_raises_when_input_list_is_3d(self) -> None:
        network = Network(
            input_count=2,
            layer_sizes=[1],
            activation_functions=[lambda x: x],
            activation_derivatives=[lambda _: 1.0],
        )

        bad_inputs = np.zeros((2, 2, 2), dtype=float)
        with self.assertRaisesRegex(ValueError, "1D sample or a 2D array"):
            network.predict(bad_inputs)


if __name__ == "__main__":
    unittest.main()
