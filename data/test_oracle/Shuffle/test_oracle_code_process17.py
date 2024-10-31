import unittest
from unittest.mock import patch

from torch import nn

from models.Shufflenetv2.shufflenetv2 import ShuffleNetV2
from mut.replace_activation import replace_activations


class TestReplaceActivations(unittest.TestCase):

    @patch('mut.replace_activations.random')
    def test_replace_activations(self, mock_random):
        # Mocking the random choice function to always return the same activation for consistency
        mock_random.choice.return_value = nn.ReLU()

        model = ShuffleNetV2()

        # Call the function to be tested
        new_model = replace_activations(model)

        # Assert that the model has been modified correctly
        self.assertIsInstance(new_model.conv, nn.ReLU)

        # Assert that all ReLU instances have been replaced
        for _, module in new_model.named_modules():
            if isinstance(module, (nn.ReLU, nn.LeakyReLU, nn.Sigmoid, nn.Tanh, nn.ELU, nn.PReLU, nn.SELU, nn.GELU)):
                self.assertIsInstance(module, nn.ReLU)

if __name__ == '__main__':
    unittest.main()
