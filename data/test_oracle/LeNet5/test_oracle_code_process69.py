import unittest
from unittest.mock import patch

import torch
from torch import nn

from models.LeNet5.model_lenet5 import LeNet5
from mut.replace_activation import replace_activations


class TestModelMutations(unittest.TestCase):

    @patch('torch.nn.modules.activation.ReLU', side_effect=lambda: nn.ReLU())
    @patch('torch.nn.modules.activation.LeakyReLU', side_effect=lambda: nn.LeakyReLU())
    @patch('torch.nn.modules.activation.Sigmoid', side_effect=lambda: nn.Sigmoid())
    @patch('torch.nn.modules.activation.Tanh', side_effect=lambda: nn.Tanh())
    @patch('torch.nn.modules.activation.ELU', side_effect=lambda: nn.ELU())
    @patch('torch.nn.modules.activation.PReLU', side_effect=lambda: nn.PReLU())
    @patch('torch.nn.modules.activation.SELU', side_effect=lambda: nn.SELU())
    @patch('torch.nn.modules.activation.GELU', side_effect=lambda: nn.GELU())
    def test_replace_activations(self, *args):
        original_model = LeNet5()
        mutated_model = replace_activations(original_model)
        # Check that at least one activation has been replaced
        self.assertNotEqual(original_model, mutated_model)

    def test_model_functionality(self):
        original_model = LeNet5()
        input_data = torch.randn(1, 1, 32, 32)
        output_before = original_model(input_data)
        
        mutated_model = replace_activations(original_model)
        output_after = mutated_model(input_data)
        
        # Check that the model outputs have changed after mutation
        self.assertFalse(torch.allclose(output_before, output_after))

if __name__ == '__main__':
    unittest.main()
