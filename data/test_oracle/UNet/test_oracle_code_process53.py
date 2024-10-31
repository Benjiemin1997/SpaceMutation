import unittest
from unittest.mock import patch

from torch import nn

from models.UNet.model_unet import UNet
from mut.remove_activation import remove_activations


class TestRemoveActivations(unittest.TestCase):

    def setUp(self):
        self.model = UNet() # Initialize your model here



    def test_remove_activations(self, mock_double_conv):
        original_model = self.model.state_dict().copy()
        modified_model = remove_activations(self.model)

        # Check if any ReLU, LeakyReLU, Sigmoid, or Tanh modules have been removed
        for name, module in modified_model.named_modules():
            if isinstance(module, (nn.ReLU, nn.LeakyReLU, nn.Sigmoid, nn.Tanh)):
                self.fail(f"Activation {name} was not removed as expected.")

        # Check if the model state_dict has changed
        self.assertNotEqual(self.model.state_dict(), original_model)

        # Check if the expected model matches the modified model
        self.assertEqual(modified_model.state_dict(), self.expected_model.state_dict())

if __name__ == '__main__':
    unittest.main()

