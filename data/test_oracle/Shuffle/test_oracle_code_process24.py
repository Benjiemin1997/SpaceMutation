import unittest
from unittest.mock import patch

from torch import nn

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
    def test_replace_activations(self, gelu_mock, selu_mock, prelu_mock, elu_mock, tanh_mock, sigmoid_mock, leakyrelu_mock, relu_mock):

        # Create a mock model
        model = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=(3, 3)),
            nn.BatchNorm2d(24),
            nn.ReLU(),
            nn.Conv2d(24, 58, kernel_size=(1, 1)),
            nn.BatchNorm2d(58),
            nn.ReLU()
        )

        # Mock the activation functions
        relu_mock.return_value = nn.ReLU()
        leakyrelu_mock.return_value = nn.LeakyReLU()
        sigmoid_mock.return_value = nn.Sigmoid()
        tanh_mock.return_value = nn.Tanh()
        elu_mock.return_value = nn.ELU()
        prelu_mock.return_value = nn.PReLU()
        selu_mock.return_value = nn.SELU()
        gelu_mock.return_value = nn.GELU()

        # Apply the mutation operator
        mutated_model = replace_activations(model)

        # Check that the original model is unchanged
        self.assertEqual(id(model), id(mutated_model))

        # Check that all activations have been replaced
        for name, module in mutated_model.named_modules():
            if isinstance(module, (nn.ReLU, nn.LeakyReLU, nn.Sigmoid, nn.Tanh, nn.ELU, nn.PReLU, nn.SELU, nn.GELU)):
                self.assertIn(name, ['relu', 'leakyrelu', 'sigmoid', 'tanh', 'elu', 'prelu', 'selu', 'gelu'])

if __name__ == '__main__':
    unittest.main()
