import unittest
from unittest.mock import patch, Mock

from torch import nn

from mut.replace_activation import replace_activations


class TestReplaceActivations(unittest.TestCase):

    @patch('your_module.random.choice', side_effect=[Mock(), Mock()])
    @patch('your_module.nn.ReLU', side_effect=[Mock(), Mock()])
    @patch('your_module.nn.LeakyReLU', side_effect=[Mock(), Mock()])
    @patch('your_module.nn.Sigmoid', side_effect=[Mock(), Mock()])
    @patch('your_module.nn.Tanh', side_effect=[Mock(), Mock()])
    @patch('your_module.nn.ELU', side_effect=[Mock(), Mock()])
    @patch('your_module.nn.PReLU', side_effect=[Mock(), Mock()])
    @patch('your_module.nn.SELU', side_effect=[Mock(), Mock()])
    @patch('your_module.nn.GELU', side_effect=[Mock(), Mock()])
    def test_replace_activations(self, *args):
        # Prepare a mock model
        model_mock = Mock()
        # Call your function under test
        result_model = replace_activations(model_mock)
        # Assert that the model's modules have been replaced
        self.assertEqual(len(result_model.modules()), len(model_mock.modules()))
        # Iterate over all modules and check that they were replaced by an activation function
        for i, (old_module, new_module) in enumerate(zip(model_mock.modules(), result_model.modules())):
            if isinstance(old_module, (nn.ReLU, nn.LeakyReLU, nn.Sigmoid, nn.Tanh, nn.ELU, nn.PReLU, nn.SELU, nn.GELU)):
                self.assertIsInstance(new_module, (nn.ReLU, nn.LeakyReLU, nn.Sigmoid, nn.Tanh, nn.ELU, nn.PReLU, nn.SELU, nn.GELU))
        # Check that the model was returned correctly
        self.assertIs(result_model, model_mock)

if __name__ == '__main__':
    unittest.main()