import unittest

import numpy as np

from multi_layer_perceptron.network import Network


class HigherLevelApiTest(unittest.TestCase):
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

        self.assertEqual(len(history), len(inputs) * 5)
        self.assertEqual(predictions.shape, targets.shape)
        self.assertGreaterEqual(mse, 0.0)


if __name__ == "__main__":
    unittest.main()
