import unittest

from models.Shufflenetv2.shufflenetv2 import ShuffleNetV2
from mut.remove_activation import remove_activations


class TestRemoveActivations(unittest.TestCase):

    def test_remove_activations(self):
        # Create a sample model
        model = ShuffleNetV2()

        # Apply the remove_activations function
        modified_model = remove_activations(model)

        # Assert that the number of modules has changed
        self.assertNotEqual(len(modified_model.modules()), len(model.modules()))

        # Check that the activation layers have been removed or replaced
        for name, module in modified_model.named_modules():
            self.assertNotIsInstance(module, (nn.ReLU, nn.LeakyReLU, nn.Sigmoid, nn.Tanh))

        # Additional assertions can be added based on expected behavior after removing activations

if __name__ == '__main__':
    unittest.main()
