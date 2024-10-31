import unittest

import torch
from torch import nn

from models.VGG16.model_vgg16 import VGG16
from mut.replace_activation import replace_activations

class TestReplaceActivations(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.model = VGG16()


    def test_replace_activations(self):

        mutated_model = replace_activations(self.model)


        for name, module in mutated_model.named_modules():
            if 'activation' in name:
                self.assertNotEqual(type(module), nn.ReLU)
                self.assertNotEqual(type(module), nn.LeakyReLU)


        input_data = torch.randn(1, 3, 224, 224)
        output_before = self.model(input_data)
        output_after = mutated_model(input_data)

        # Check if the outputs are significantly different due to the change in activation functions
        self.assertNotEqual(torch.sum(torch.abs(output_before - output_after)), 0)

if __name__ == '__main__':
    unittest.main()
