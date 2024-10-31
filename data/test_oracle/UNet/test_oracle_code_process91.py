import unittest
from unittest.mock import patch, MagicMock
from mut.replace_activation import replace_activations
from torch import nn

class TestReplaceActivations(unittest.TestCase):

    @patch('torch.nn.Module', autospec=True)
    def test_replace_activations(self, mock_model):
        # Create a mock model instance
        mock_model_instance = MagicMock(spec=nn.Module)
        mock_model_instance.named_modules.return_value = [
            ('downs.0.conv.0.0', nn.ReLU()),
            ('downs.0.conv.1', nn.BatchNorm2d(64)),
            ('downs.0.conv.2', nn.Identity()),
            ('downs.0.conv.3.0', nn.LeakyReLU()),
            ('downs.0.conv.4', nn.BatchNorm2d(64)),
            ('downs.0.conv.5', nn.Identity()),
            ('downs.1.conv.0', nn.ReLU()),
            ('downs.1.conv.1', nn.BatchNorm2d(128)),
            ('downs.1.conv.2', nn.ELU(alpha=1.0)),
            ('downs.1.conv.3', nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))),
            ('downs.1.conv.4', nn.BatchNorm2d(128)),
            ('downs.1.conv.5', nn.Identity())
        ]

        # Apply the replace_activations function to the mock model
        modified_model = replace_activations(mock_model_instance)

        # Check that all ReLU activations have been replaced
        for _, module in mock_model_instance.named_modules():
            if isinstance(module, nn.ReLU):
                self.assertNotEqual(type(module), nn.ReLU)
                self.assertIn(type(module), [nn.LeakyReLU, nn.Sigmoid, nn.Tanh, nn.ELU, nn.PReLU, nn.SELU, nn.GELU])

        # Check that the original model is unchanged
        self.assertIs(mock_model.return_value, mock_model_instance)

if __name__ == '__main__':
    unittest.main()

