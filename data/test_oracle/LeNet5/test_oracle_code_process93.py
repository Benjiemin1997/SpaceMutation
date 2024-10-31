import unittest
from unittest.mock import patch, Mock

from torch import nn

from models.LeNet5.model_lenet5 import LeNet5
from mut.replace_activation import replace_activations


class TestReplaceActivations(unittest.TestCase):

    @patch('torch.nn.modules.activation.ReLU', side_effect=lambda: nn.ReLU())
    @patch('torch.nn.modules.activation.LeakyReLU', side_effect=lambda: nn.LeakyReLU())
    @patch('torch.nn.modules.activation.Sigmoid', side_effect=lambda: nn.Sigmoid())
    @patch('torch.nn.modules.activation.Tanh', side_effect=lambda: nn.Tanh())
    @patch('torch.nn.modules.activation.ELU', side_effect=lambda: nn.ELU())
    @patch('torch.nn.modules.activation.PReLU', side_effect=lambda: nn.PReLU())
    @patch('torch.nn.modules.activation.SELU', side_effect=lambda: nn.SELU())
    @patch('torch.nn.modules.activation.GELU', side_effect=lambda: nn.GELU())
    def test_replace_activations(self, *args):
        model = LeNet5()
        replace_activations(model)
        
        # Check if all ReLU, LeakyReLU, Sigmoid, Tanh, ELU, PReLU, SELU, GELU instances were replaced
        self.assertTrue(all(isinstance(m, nn.ReLU) for m in model.modules() if isinstance(m, nn.Module)))
        self.assertTrue(all(isinstance(m, nn.LeakyReLU) for m in model.modules() if isinstance(m, nn.Module)))
        self.assertTrue(all(isinstance(m, nn.Sigmoid) for m in model.modules() if isinstance(m, nn.Module)))
        self.assertTrue(all(isinstance(m, nn.Tanh) for m in model.modules() if isinstance(m, nn.Module)))
        self.assertTrue(all(isinstance(m, nn.ELU) for m in model.modules() if isinstance(m, nn.Module)))
        self.assertTrue(all(isinstance(m, nn.PReLU) for m in model.modules() if isinstance(m, nn.Module)))
        self.assertTrue(all(isinstance(m, nn.SELU) for m in model.modules() if isinstance(m, nn.Module)))
        self.assertTrue(all(isinstance(m, nn.GELU) for m in model.modules() if isinstance(m, nn.Module)))

if __name__ == '__main__':
    unittest.main()
