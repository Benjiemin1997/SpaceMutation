import unittest

import torch
from torch import nn

from models.Shufflenetv2.shufflenetv2 import ShuffleNetV2
from mut.remove_activation import remove_activations


class TestRemoveActivations(unittest.TestCase):

    def test_remove_activations(self):
        # Create a sample model
        model = ShuffleNetV2()

        # Check the original model has activation functions
        self.assertTrue(any(isinstance(module, (nn.ReLU, nn.LeakyReLU, nn.Sigmoid, nn.Tanh)) for _, module in model.named_modules()))

        # Apply the remove_activations function
        new_model = remove_activations(model)

        # Check that some activation functions have been removed
        self.assertFalse(any(isinstance(module, (nn.ReLU, nn.LeakyReLU, nn.Sigmoid, nn.Tanh)) for _, module in new_model.named_modules()))

        # Check that the model still works after removing activations
        input_data = torch.randn(1, 3, 224, 224)
        output_before = model(input_data)
        output_after = new_model(input_data)
        self.assertEqual(output_before.shape, output_after.shape)
        self.assertTrue(torch.allclose(output_before, output_after))

if __name__ == '__main__':
    unittest.main()
