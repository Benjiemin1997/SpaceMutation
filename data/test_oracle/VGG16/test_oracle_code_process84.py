import unittest
from unittest.mock import patch, MagicMock

from torch import nn

from mut.replace_activation import replace_activations
from torchvision.models import vgg16
import torch

class TestReplaceActivations(unittest.TestCase):

    @patch('torch.nn.modules.activation.ReLU')
    @patch('torch.nn.modules.activation.LeakyReLU')
    @patch('torch.nn.modules.activation.Sigmoid')
    @patch('torch.nn.modules.activation.Tanh')
    @patch('torch.nn.modules.activation.ELU')
    @patch('torch.nn.modules.activation.PReLU')
    @patch('torch.nn.modules.activation.SELU')
    @patch('torch.nn.modules.activation.GELU')
    def test_replace_activations(self, mock_GELU, mock_SELU, mock_PReLU, mock_ELU, mock_Tanh, mock_Sigmoid, mock_LeakyReLU, mock_ReLU):

        model = vgg16(pretrained=True)
        

        mock_ReLU.return_value = 'Mocked ReLU'
        mock_LeakyReLU.return_value = 'Mocked LeakyReLU'
        mock_Sigmoid.return_value = 'Mocked Sigmoid'
        mock_Tanh.return_value = 'Mocked Tanh'
        mock_ELU.return_value = 'Mocked ELU'
        mock_PReLU.return_value = 'Mocked PReLU'
        mock_SELU.return_value = 'Mocked SELU'
        mock_GELU.return_value = 'Mocked GELU'
        
        # Replace activations in the model
        replaced_model = replace_activations(model)
        
        # Check that each activation has been replaced
        for name, module in replaced_model.named_modules():
            if isinstance(module, (nn.ReLU, nn.LeakyReLU, nn.Sigmoid, nn.Tanh, nn.ELU, nn.PReLU, nn.SELU, nn.GELU)):
                self.assertIn(name, ['Mocked ReLU', 'Mocked LeakyReLU', 'Mocked Sigmoid', 'Mocked Tanh', 'Mocked ELU', 'Mocked PReLU', 'Mocked SELU', 'Mocked GELU'])
                
if __name__ == '__main__':
    unittest.main()
