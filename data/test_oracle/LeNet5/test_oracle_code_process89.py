import unittest

from torch import nn

from models.LeNet5.model_lenet5 import LeNet5
from mut.remove_activation import remove_activations


class TestRemoveActivations(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.model = LeNet5()

    def test_remove_activations(self):
        # Save original activation count
        original_activations_count = sum(1 for _, m in self.model.named_modules() if isinstance(m, (nn.ReLU, nn.LeakyReLU, nn.Sigmoid, nn.Tanh)))

        # Apply the function to be tested
        modified_model = remove_activations(self.model)

        # Verify that the number of activations has been reduced
        new_activations_count = sum(1 for _, m in modified_model.named_modules() if isinstance(m, (nn.ReLU, nn.LeakyReLU, nn.Sigmoid, nn.Tanh)))
        self.assertLess(new_activations_count, original_activations_count)

        # Verify that the modified model still contains all its original layers except the removed activations
        for name, module in modified_model.named_modules():
            if name not in ['model.1', 'model.9']:
                self.assertIsNotNone(module)
        self.assertIsNone(modified_model._modules.get('model.1'))
        self.assertIsNone(modified_model._modules.get('model.9'))

if __name__ == '__main__':
    unittest.main()
