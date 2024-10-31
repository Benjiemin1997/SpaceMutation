import unittest

import torch
from torch import nn

from models.VGG16.model_vgg16 import VGG16
from mut.random_add_activation import add_activation


class TestAddActivation(unittest.TestCase):
    
    def setUp(self):
        self.vgg_model = VGG16()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_add_activation(self):
        mutated_model = add_activation(self.vgg_model)
        for name, module in mutated_model.named_modules():
            if 'Sequential' in str(type(module)):
                self.assertTrue(len(list(module.children())) > 1,
                                "Model was not mutated properly.")

        for name, module in mutated_model.named_modules():
            if isinstance(module, nn.Sequential):
                last_module = list(module.children())[-1]
                self.assertIn(type(last_module), 
                              [nn.ReLU, nn.LeakyReLU, nn.Sigmoid, nn.Tanh, nn.ELU, nn.PReLU, nn.SELU, nn.GELU],
                              f"Last module in sequential is not an expected activation function type. Found: {type(last_module)}")

if __name__ == '__main__':
    unittest.main()
