import unittest
from unittest.mock import patch

from torch import nn

from models.LeNet5.model_lenet5 import LeNet5
from mut.remove_activation import remove_activations


class TestRemoveActivations(unittest.TestCase):

    @patch('torch.nn.ReLU', side_effect=lambda: nn.ReLU())
    @patch('torch.nn.LeakyReLU', side_effect=lambda: nn.LeakyReLU())
    @patch('torch.nn.Sigmoid', side_effect=lambda: nn.Sigmoid())
    @patch('torch.nn.Tanh', side_effect=lambda: nn.Tanh())
    def test_remove_activations(self, relu_mock, leaky_relu_mock, sigmoid_mock, tanh_mock):
        model = LeNet5()
        modified_model = remove_activations(model)

        # Check that activation functions have been removed
        for name, module in modified_model.named_modules():
            self.assertNotIn(name, ['ReLU', 'LeakyReLU', 'Sigmoid', 'Tanh'])

        # Check that nn.Identity() has been inserted instead of activation functions
        identity_count = sum([1 for name, module in modified_model.named_modules() if isinstance(module, nn.Identity)])
        self.assertEqual(identity_count, len(relu_mock.call_args_list) + len(leaky_relu_mock.call_args_list) + len(sigmoid_mock.call_args_list) + len(tanh_mock.call_args_list))

if __name__ == '__main__':
    unittest.main()
