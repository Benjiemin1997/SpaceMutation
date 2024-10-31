import unittest
from unittest.mock import patch

from torch import nn

from mut.replace_activation import replace_activations


class TestReplaceActivations(unittest.TestCase):

    def setUp(self):
        # Create a mock model for testing
        self.model = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=(3, 3)),
            nn.ReLU(),
            nn.Conv2d(24, 58, kernel_size=(1, 1)),
            nn.Sigmoid(),
            nn.Conv2d(58, 116, kernel_size=(1, 1)),
            nn.ELU(),
            nn.Linear(116, 100),
            nn.Tanh()
        )

    @patch('torch.nn.modules.activation.ReLU')
    @patch('torch.nn.modules.activation.LeakyReLU')
    @patch('torch.nn.modules.activation.Sigmoid')
    @patch('torch.nn.modules.activation.Tanh')
    @patch('torch.nn.modules.activation.ELU')
    @patch('torch.nn.modules.activation.PReLU')
    @patch('torch.nn.modules.activation.SELU')
    @patch('torch.nn.modules.activation.GELU')
    def test_replace_activations(self, gelu_mock, selu_mock, prelu_mock, elu_mock, tanh_mock, sigmoid_mock, leaky_relu_mock, relu_mock):
        # Replace activations with random ones
        replaced_model = replace_activations(self.model)

        # Check that all ReLUs have been replaced
        self.assertNotEqual(self.model[1], replaced_model[1])
        self.assertIsInstance(replaced_model[1], gelu_mock.return_value)
        
        # Check that all LeakyREUs have been replaced
        self.assertNotEqual(self.model[3], replaced_model[3])
        self.assertIsInstance(replaced_model[3], selu_mock.return_value)
        
        # Check that all Sigmoids have been replaced
        self.assertNotEqual(self.model[5], replaced_model[5])
        self.assertIsInstance(replaced_model[5], prelu_mock.return_value)
        
        # Check that all Tans have been replaced
        self.assertNotEqual(self.model[7], replaced_model[7])
        self.assertIsInstance(replaced_model[7], elu_mock.return_value)
        
        # Check that all SELUs have been replaced
        self.assertNotEqual(self.model[9], replaced_model[9])
        self.assertIsInstance(replaced_model[9], tanh_mock.return_value)
        
        # Check that all PRELUs have been replaced
        self.assertNotEqual(self.model[11], replaced_model[11])
        self.assertIsInstance(replaced_model[11], leaky_relu_mock.return_value)
        
        # Check that all RELUs have been replaced
        self.assertNotEqual(self.model[13], replaced_model[13])
        self.assertIsInstance(replaced_model[13], relu_mock.return_value)

if __name__ == '__main__':
    unittest.main()
