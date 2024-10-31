import unittest

from models.UNet.model_unet import UNet
from mut.remove_activation import remove_activations


class TestRemoveActivations(unittest.TestCase):

    def setUp(self):
        self.model = UNet()

    def test_remove_activations(self):
        remove_activations(self.model)

        for name, module in self.model.named_modules():
            if 'ReLU' in str(type(module)):
                self.assertIsNone(module)

if __name__ == '__main__':
    unittest.main()
