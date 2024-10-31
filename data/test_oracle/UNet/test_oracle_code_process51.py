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
        original_model = self.model.state_dict().copy()
        
        # Apply reverse activation transformation
        reverse_activations(self.model)
        
        # Check if inplace flag is False for all ReLUs
        for name, module in self.model.named_children():
            if isinstance(module, nn.ReLU):
                self.assertFalse(hasattr(module, 'inplace'))
        
        # Check if forward hooks were registered properly
        for name, module in self.model.named_children():
            if isinstance(module, nn.ReLU):
                prev_module = list(self.model.named_children())[list(self.model.named_children()).index((name, module)) - 1][1]
                self.assertTrue(hasattr(prev_module, 'register_forward_hook'))
        
        # Compare model states before and after transformation
        for key in original_model:
            self.assertTrue(torch.equal(original_model[key], self.model.state_dict()[key]), f"Model state {key} mismatch")
            
        # Run a forward pass to ensure no errors occur
        dummy_input = torch.randn(1, 1, 32, 32)
        _ = self.model(dummy_input)

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
