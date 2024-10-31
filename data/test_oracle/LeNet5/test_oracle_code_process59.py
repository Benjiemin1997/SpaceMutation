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

        self.assertEqual(len(result_model.modules()), len(model_mock.modules()))
        # Iterate over all modules and check that they were replaced with an activation layer
        for idx, module in enumerate(result_model.modules()):
            original_module = model_mock.modules()[idx]
            if isinstance(original_module, (nn.ReLU, nn.LeakyReLU, nn.Sigmoid, nn.Tanh, nn.ELU, nn.PReLU, nn.SELU, nn.GELU)):
                self.assertIsInstance(module, (nn.ReLU, nn.LeakyReLU, nn.Sigmoid, nn.Tanh, nn.ELU, nn.PReLU, nn.SELU, nn.GELU))

if __name__ == '__main__':
    unittest.main()
