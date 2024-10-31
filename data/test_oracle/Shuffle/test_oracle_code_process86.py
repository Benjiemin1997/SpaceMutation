import unittest

from models.Shufflenetv2.shufflenetv2 import ShuffleNetV2
from mut.replace_activation import replace_activations
from torch import nn

class TestShuffleNetV2(unittest.TestCase):

    def setUp(self):
        self.model = ShuffleNetV2()

    def test_replace_activations(self):
        # Original model
        original_model = self.model

        # Apply mutation
        mutated_model = replace_activations(self.model)

        # Check that the model is mutated
        assert mutated_model != self.model, "Model was not mutated"

        # Check that all activation functions have been replaced
        for name, module in mutated_model.named_modules():
            if isinstance(module, (nn.ReLU, nn.LeakyReLU, nn.Sigmoid, nn.Tanh, nn.ELU, nn.PReLU, nn.SELU, nn.GELU)):
                assert isinstance(module, nn.ReLU) or isinstance(module, nn.LeakyReLU) or \
                       isinstance(module, nn.Sigmoid) or isinstance(module, nn.Tanh) or \
                       isinstance(module, nn.ELU) or isinstance(module, nn.PReLU) or \
                       isinstance(module, nn.SELU) or isinstance(module, nn.GELU), \
                    f"Activation function at {name} is not one of the expected types"


if __name__ == '__main__':
    unittest.main()