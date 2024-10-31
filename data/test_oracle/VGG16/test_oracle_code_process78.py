import unittest
from unittest.mock import patch, Mock

from torch import nn

from mut.replace_activation import replace_activations
from torchvision.models import vgg16
import torch

class TestReplaceActivations(unittest.TestCase):

    @patch('torch.nn.Module')
    def test_replace_activations(self, mock_module):

        mock_model = Mock(spec=vgg16(pretrained=True))
        mock_module.return_value = mock_model


        mutated_model = replace_activations(mock_model)


        for name, module in mutated_model.named_modules():
            if isinstance(module, (nn.ReLU, nn.LeakyReLU, nn.Sigmoid, nn.Tanh, nn.ELU, nn.PReLU, nn.SELU, nn.GELU)):
                self.assertIn(type(module), [nn.ReLU, nn.LeakyReLU, nn.Sigmoid, nn.Tanh, nn.ELU, nn.PReLU, nn.SELU, nn.GELU])

if __name__ == '__main__':
    unittest.main()
