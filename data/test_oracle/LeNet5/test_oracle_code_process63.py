import unittest
from unittest.mock import patch

import torch
from torch import nn

from models.LeNet5.model_lenet5 import LeNet5
from mut.remove_activation import remove_activations


class TestRemoveActivations(unittest.TestCase):

    @patch('torch.cuda.is_available', return_value=True)
    def setUp(self, mock_cuda_available):
        self.model = LeNet5()
        self.model.eval()

    def test_remove_activations(self):
        original_model = self.model.state_dict().copy()
        modified_model = remove_activations(self.model)
        
        # Check that any ReLU, LeakyReLU, Sigmoid, or Tanh layers have been replaced with Identity layers
        for name, module in modified_model.named_modules():
            if isinstance(module, (nn.ReLU, nn.LeakyReLU, nn.Sigmoid, nn.Tanh)):
                self.assertIsInstance(module, nn.Identity)
        
        # Ensure that the model's output remains consistent after removing activations
        input_data = torch.randn(1, 1, 32, 32)
        original_output = self.model(input_data)
        modified_output = modified_model(input_data)
        self.assertTrue(torch.allclose(original_output, modified_output, atol=1e-3))

        # Ensure that the model state dict has been altered
        self.assertNotEqual(self.model.state_dict(), original_model)

if __name__ == '__main__':
    unittest.main()