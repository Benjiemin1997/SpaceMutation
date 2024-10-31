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
                last_layer = list(module.children())[-1]
                self.assertTrue(isinstance(last_layer, nn.Module), "Last layer should be an activation layer after mutation")

if __name__ == '__main__':
    unittest.main()
