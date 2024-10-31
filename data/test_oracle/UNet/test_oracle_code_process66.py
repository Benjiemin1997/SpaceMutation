import unittest
from unittest.mock import patch

from torch import nn

from models.UNet.model_unet import UNet
from mut.remove_activation import remove_activations


class TestRemoveActivations(unittest.TestCase):

    def setUp(self):
        self.model = UNet()

    def test_remove_activations(self, mock_choice):
        removed_model = remove_activations(self.model)
        self.assertEqual(removed_model.downs[0].conv[0][1], nn.Identity(), "Activation was not removed correctly")

        # Add more assertions as needed based on your model's structure and expected behavior after activation removal

if __name__ == '__main__':
    unittest.main()
