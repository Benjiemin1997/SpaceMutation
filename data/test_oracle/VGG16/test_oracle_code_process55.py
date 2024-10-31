import unittest

import torch
from torch import nn

from models.VGG16.model_vgg16 import VGG16
from mut.random_add_activation import add_activation


class TestAddActivation(unittest.TestCase):
    
    def setUp(self):
        self.model = VGG16()
        self.model = add_activation(self.model)
        
    def test_model_structure(self):
        activation_types = [type(nn.ReLU()), type(nn.LeakyReLU()), type(nn.Sigmoid()), type(nn.Tanh()),
                            type(nn.ELU()), type(nn.PReLU()), type(nn.SELU()), type(nn.GELU())]
        for name, module in self.model.named_modules():
            for child_name, child in module.named_children():
                if isinstance(child, nn.Sequential) and len(child) == 2:
                    _, last_child = child
                    self.assertIn(type(last_child), activation_types, f"Activation {type(last_child)} not found in {name}.{child_name}")
                    
    def test_forward_pass(self):
        input_data = torch.randn(1, 3, 32, 32)
        output_before = self.model(input_data)
        self.model = add_activation(self.model)
        output_after = self.model(input_data)
        self.assertTrue(torch.allclose(output_before, output_after, atol=1e-4, rtol=1e-4), "Forward pass output changed after adding activation")

if __name__ == '__main__':
    unittest.main()
