import unittest
from torch import nn

from models.UNet.model_unet import UNet
from mut.remove_activation import remove_activations


class TestRemoveActivations(unittest.TestCase):

    def setUp(self):
        self.model = UNet()  # Initialize your model here

    def test_remove_activations(self, mock_gaussian_fuzz, mock_random_shuffle, mock_replace_activations, mock_selu,
                                mock_gelu, mock_prelu, mock_batchnorm, mock_conv, mock_doubleconv):
        # Call the function you want to test
        modified_model = remove_activations(self.model)

        # Assertions based on expected behavior
        self.assertEqual(len(modified_model.downs), len(self.expected_model.downs))
        self.assertEqual(len(modified_model.ups), len(self.expected_model.ups))

        for name, module in modified_model.named_modules():
            if isinstance(module, (nn.ReLU, nn.LeakyReLU, nn.Sigmoid, nn.Tanh)):
                self.assertIsNone(getattr(modified_model, name))

        # You can add more specific assertions here based on your model structure and expected changes after activation removal.

if __name__ == '__main__':
    unittest.main()
