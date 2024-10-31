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

        num_linear_or_conv_layers = sum([isinstance(m, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)) for m in self.model.modules()])
        num_activation_layers = sum([isinstance(m, nn.Module) and len(list(m.children())) > 0 for m in self.model.modules()])
        
        self.assertGreater(num_activation_layers, 0, "No activation layers added to the model.")
        self.assertLess(num_activation_layers, num_linear_or_conv_layers, "More than one activation layer was added per linear or conv layer.")
        
    def test_forward_pass(self):

        input_data = torch.randn(1, 3, 32, 32)
        output_before = self.model(input_data)
        
        self.model.eval()
        output_after = self.model(input_data)

        self.assertFalse(torch.allclose(output_before, output_after), "Output after adding activation is the same as before.")
        
if __name__ == '__main__':
    unittest.main()
