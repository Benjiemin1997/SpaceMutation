import unittest

import torch
from torch import nn

from models.Shufflenetv2.shufflenetv2 import ShuffleNetV2
from mut.remove_activation import remove_activations


class TestRemoveActivations(unittest.TestCase):

    def test_remove_activations(self):
        # Create a sample model
        model = ShuffleNetV2()

        # Check the original model has some activation functions
        original_activations = sum([isinstance(m, (nn.ReLU, nn.LeakyReLU, nn.Sigmoid, nn.Tanh)) for m in model.modules()])
        self.assertGreater(original_activations, 0)

        # Apply the remove_activations function
        modified_model = remove_activations(model)

        # Check that at least one activation function was removed
        after_activations = sum([isinstance(m, (nn.ReLU, nn.LeakyReLU, nn.Sigmoid, nn.Tanh)) for m in modified_model.modules()])
        self.assertLessEqual(after_activations, original_activations - 1)

        # Check that the model still works after removing activations
        input_tensor = torch.randn(1, 3, 224, 224)
        output_original = model(input_tensor)
        output_modified = modified_model(input_tensor)
        self.assertEqual(output_original.shape, output_modified.shape)

if __name__ == '__main__':
    unittest.main()