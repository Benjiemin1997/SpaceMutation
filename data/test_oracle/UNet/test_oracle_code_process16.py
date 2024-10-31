import torch
import unittest

from torch import nn

from models.UNet.model_unet import UNet
from mut.reverse_activation import reverse_activations


class TestReverseActivations(unittest.TestCase):

    def setUp(self):
        # Create a simple test model
        self.model = UNet()

    def test_reverse_activations(self):
        # Check if inplace is set to False after calling reverse_activations
        reverse_activations(self.model)
        for _, module in self.model.named_modules():
            if isinstance(module, nn.ReLU):
                self.assertFalse(hasattr(module, 'inplace'))

        # Check if forward hooks are registered
        input = torch.randn(1, 1, 10, 10)
        output = self.model(input)
        prev_module = None
        for name, module in self.model.named_modules():
            if isinstance(module, nn.ReLU):
                prev_module = prev_module or list(self.model.modules())[list(self.model.modules()).index(name) - 1]
                self.assertTrue(prev_module.has_forward_hook)
                prev_module.forward_hook.remove()

        # Check if output is negated as expected
        output_negated = self.model(input)
        self.assertTrue(torch.allclose(-output, output_negated))

if __name__ == '__main__':
    unittest.main()
