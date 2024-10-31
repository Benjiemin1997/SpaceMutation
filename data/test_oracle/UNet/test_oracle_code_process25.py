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
            ('downs.0.conv.3.1', nn.Identity()),
            ('downs.0.conv.4', nn.BatchNorm2d(64)),
            ('downs.0.conv.5', nn.Identity())
        ]

        # Replace activations
        new_model = replace_activations(mock_model_instance)

        # Check if the activation was replaced
        self.assertNotEqual(new_model.downs[0].conv[3], nn.ReLU())

        # Check if the other activations remained unchanged
        self.assertIsInstance(new_model.downs[0].conv[0], nn.ReLU)
        self.assertIsInstance(new_model.downs[0].conv[1], nn.BatchNorm2d)
        self.assertIsInstance(new_model.downs[0].conv[2], nn.Identity)
        self.assertIsInstance(new_model.downs[0].conv[4], nn.BatchNorm2d)
        self.assertIsInstance(new_model.downs[0].conv[5], nn.Identity)

if __name__ == '__main__':
    unittest.main()
