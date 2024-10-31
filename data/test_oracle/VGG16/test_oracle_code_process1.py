import unittest
from unittest.mock import patch, MagicMock

from torch import nn

from mut.replace_activation import replace_activations
from torchvision.models import vgg16

class TestReplaceActivations(unittest.TestCase):

    @patch('torch.nn.Module')
    def test_replace_activations(self, mock_module):
        # Create a mock VGG16 model
        mock_vgg16 = MagicMock(spec=vgg16(pretrained=True))
        mock_module.return_value = mock_vgg16
        
        # Replace activation functions randomly in the mock VGG16 model
        replace_activations(mock_vgg16)

        # Assert that each activation has been replaced
        for name, module in mock_vgg16.named_modules():
            if isinstance(module, (nn.ReLU, nn.LeakyReLU, nn.Sigmoid, nn.Tanh, nn.ELU, nn.PReLU, nn.SELU, nn.GELU)):
                self.assertNotEqual(type(module), type(module))

if __name__ == '__main__':
    unittest.main()
