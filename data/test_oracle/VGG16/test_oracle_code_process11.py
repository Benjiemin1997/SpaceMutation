import random
import unittest
from unittest.mock import patch

from torch import nn

from models.VGG16.model_vgg16 import VGG16
from mut.replace_activation import replace_activations


class TestReplaceActivations(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.model = VGG16()
        # Initialize your model here, possibly with some predefined weights or data

    def test_replace_activations(self):
        # Check if the activation replacement is successful
        modified_model = replace_activations(self.model)
        for name, module in modified_model.named_modules():
            if isinstance(module, (nn.ReLU, nn.LeakyReLU, nn.Sigmoid, nn.Tanh, nn.ELU, nn.PReLU, nn.SELU, nn.GELU)):
                self.assertNotEqual(type(module), type(module.activation))  # Check that the type has changed
                self.assertEqual(type(module), type(random.choice([nn.ReLU(), nn.LeakyReLU(), nn.Sigmoid(), nn.Tanh(), nn.ELU(), nn.PReLU(), nn.SELU(), nn.GELU()])))  # Check that it's one of the expected types


    def test_device(self, mock_cuda):
        # Check that the device is set to GPU if available
        modified_model = replace_activations(self.model)
        self.assertEqual(next(modified_model.parameters()).device.type, 'cuda')

    def tearDown(self):
        pass

if __name__ == '__main__':
    unittest.main()
