import unittest
from unittest.mock import patch, Mock

from torch import nn

from models.LeNet5.model_lenet5 import LeNet5
from mut.replace_activation import replace_activations


class TestReplaceActivations(unittest.TestCase):

    def test_replace_activations(self, *args):
        # Prepare a mock model
        model_mock = LeNet5()
        # Call your function under test
        result_model = replace_activations(model_mock)

        self.assertEqual(len(result_model.modules()), len(model_mock.modules()))

        for i, module in enumerate(result_model.modules()):
            original_module = model_mock.modules()[i]
            if isinstance(original_module, (nn.ReLU, nn.LeakyReLU, nn.Sigmoid, nn.Tanh, nn.ELU, nn.PReLU, nn.SELU, nn.GELU)):
                self.assertIsInstance(module, (nn.ReLU, nn.LeakyReLU, nn.Sigmoid, nn.Tanh, nn.ELU, nn.PReLU, nn.SELU, nn.GELU))

if __name__ == '__main__':
    unittest.main()
