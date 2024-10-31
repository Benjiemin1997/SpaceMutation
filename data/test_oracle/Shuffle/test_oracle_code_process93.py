import unittest
from unittest.mock import patch

from torch import nn

from models.Shufflenetv2.shufflenetv2 import ShuffleNetV2
from mut.replace_activation import replace_activations


class TestReplaceActivations(unittest.TestCase):

    @patch('torch.nn.ReLU')
    @patch('torch.nn.LeakyReLU')
    @patch('torch.nn.Sigmoid')
    @patch('torch.nn.Tanh')
    @patch('torch.nn.ELU')
    @patch('torch.nn.PReLU')
    @patch('torch.nn.SELU')
    @patch('torch.nn.GELU')
    def test_replace_activations(self, mock_GELU, mock_SELU, mock_PReLU, mock_PRelu, mock_Tanh, mock_Sigmoid, mock_LeakyReLU, mock_ReLU):
        
        # Define a mock model
        class MockModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(3, 24, kernel_size=(3, 3))
                self.relu1 = nn.ReLU()
                self.leakyrelu1 = nn.LeakyReLU()
                self.sigmoid1 = nn.Sigmoid()
                self.tanh1 = nn.Tanh()
                self.elu1 = nn.ELU()
                self.prelu1 = nn.PReLU()
                self.selu1 = nn.SELU()
                self.gelu1 = nn.GELU()

        mock_model = ShuffleNetV2()

        # Apply the replace_activations function to the mock model
        replaced_model = replace_activations(mock_model)

        # Check that each layer has been replaced with a randomly chosen activation
        for original_layer, replaced_layer in zip([mock_model.relu1, mock_model.leakyrelu1, mock_model.sigmoid1, mock_model.tanh1, mock_model.elu1, mock_model.prelu1, mock_model.selu1, mock_model.gelu1], 
                                                  [mock_ReLU(), mock_LeakyReLU(), mock_Sigmoid(), mock_Tanh(), mock_ELU(), mock_PReLU(), mock_SELU(), mock_GELU()]):
            self.assertIsNot(original_layer, replaced_layer)

        # Check that the original model structure is preserved
        self.assertEqual(mock_model.conv1, replaced_model.conv1)

if __name__ == '__main__':
    unittest.main()
