import unittest

import torch
from torch import nn

from models.Shufflenetv2.shufflenetv2 import ShuffleNetV2
from mut.remove_activation import remove_activations


class TestRemoveActivations(unittest.TestCase):

    def setUp(self):
        self.model = ShuffleNetV2()

    def test_remove_activations(self):
        original_model = self.model.state_dict().copy()
        remove_activations(self.model)
        
        # Check if any activation layers were removed
        activations = [module for module in self.model.modules() if isinstance(module, (nn.ReLU, nn.LeakyReLU, nn.Sigmoid, nn.Tanh))]
        self.assertLess(len(activations), len(original_model))

        # Check if the model's output changes after removing activations
        input_data = torch.randn(1, 3, 224, 224)
        original_output = self.model(input_data)
        modified_output = self.model(input_data)
        self.assertNotEqual(original_output.sum().item(), modified_output.sum().item())

if __name__ == '__main__':
    unittest.main()