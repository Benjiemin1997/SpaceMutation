import unittest
from unittest.mock import patch

from torch import nn

from models.UNet.model_unet import UNet
from mut.remove_activation import remove_activations


class TestRemoveActivations(unittest.TestCase):

    def setUp(self):
        self.model = UNet()  # Initialize your model here



    def test_remove_activations(self):
        # Call the function you want to test
        modified_model = remove_activations(self.model)

        # Assertions based on expected behavior
        self.assertEqual(len(modified_model.downs), len(self.expected_model.downs))
        self.assertEqual(len(modified_model.ups), len(self.expected_model.ups))
        self.assertEqual(len(modified_model.bottleneck), len(self.expected_model.bottleneck))

        # Check if the activation layers were removed as expected
        for name, module in modified_model.named_modules():
            if isinstance(module, (nn.ReLU, nn.LeakyReLU, nn.Sigmoid, nn.Tanh)):
                self.assertIsNone(getattr(modified_model, name))

        # Check if the rest of the model's structure remains intact
        for attr in ['pool', 'final_conv']:
            self.assertIsNotNone(getattr(modified_model, attr))

if __name__ == '__main__':
    unittest.main()
