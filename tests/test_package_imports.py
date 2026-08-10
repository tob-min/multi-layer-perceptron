import unittest

from multi_layer_perceptron import Network


class PackageImportTest(unittest.TestCase):
    def test_package_imports_work(self) -> None:
        self.assertTrue(callable(Network))


if __name__ == "__main__":
    unittest.main()
