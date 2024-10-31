import torch
import unittest

from models.UNet.model_unet import UNet
from mut.reverse_activation import reverse_activations


class TestReverseActivations(unittest.TestCase):

    def setUp(self):
        # Create a simple model for testing
        self.model = UNet()
        
    def test_reverse_activations(self):
        original_output = self.model(torch.randn(1, 1, 32, 32))
        
        # Reverse activations
        reversed_model = reverse_activations(self.model)
        reversed_output = reversed_model(torch.randn(1, 1, 32, 32))
        
        # Check that outputs are opposite in sign
        self.assertTrue(torch.allclose(-original_output, reversed_output, atol=1e-4))

if __name__ == '__main__':
    unittest.main()