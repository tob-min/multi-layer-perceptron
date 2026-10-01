import argparse

from typing import cast

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.utils import Bunch

from multi_layer_perceptron.activations import sigmoid, sigmoid_derivative, tanh, tanh_derivative
from multi_layer_perceptron.network import Network


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify handwritten digits with an 8x8 MLP.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs.")
    parser.add_argument("--learning-rate", type=float, default=0.05, help="Gradient descent learning rate.")
    args = parser.parse_args()

    np.random.seed(args.seed)
    
    digits = cast(Bunch, load_digits())
    
    # normalise pixel values to [0,1]
    features = digits.data / 16.0
    labels = digits.target
    
    # split data into train and test set
    train_features, test_features, train_labels, test_labels = train_test_split(
        features,
        labels,
        test_size=0.2,
        random_state=args.seed,
        stratify=labels,
    )
    
    # one hot encode target values 
    train_targets = np.eye(10)[train_labels]

    network = Network(
        input_count=64,
        # output node for each possible digit, outputting probability of being that digit
        layer_sizes=[32, 10],
        activation_functions=[tanh, sigmoid],
        activation_derivatives=[tanh_derivative, sigmoid_derivative],
    )

    network.fit(
        train_features,
        train_targets,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
    )

    predictions = np.argmax(network.predict(test_features), axis=1)
    accuracy = np.mean(predictions == test_labels)
    print(f"Dataset: scikit-learn digits (8x8), {len(test_labels)} held-out test images")
    print(f"Test accuracy: {accuracy:.2%}")


if __name__ == "__main__":
    main()