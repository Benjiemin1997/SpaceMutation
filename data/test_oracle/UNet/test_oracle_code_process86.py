import torch
import unittest

from torch import nn

from models.UNet.model_unet import UNet
from mut.reverse_activation import reverse_activations


class TestReverseActivations(unittest.TestCase):

    def setUp(self):
        # Create a simple model for testing
        self.model = UNet()
        
    def test_reverse_activations(self):
        original_output = self.model(torch.randn(1, 1, 32, 32))
        
        # Reverse activations by changing ReLU to negative output
        reverse_model = reverse_activations(self.model)
        
        # Get new output after reversing activations
        reversed_output = reverse_model(torch.randn(1, 1, 32, 32))
        
        # Check that outputs are negatives of each other
        self.assertTrue(torch.allclose(-original_output, reversed_output))

        # Check that the model state has been updated correctly
        for name, module in reverse_model.named_children():
            if isinstance(module, nn.ReLU):
                self.assertFalse(hasattr(module, 'inplace'))
    
    def tearDown(self):
        del self.model

if __name__ == '__main__':
    unittest.main()

