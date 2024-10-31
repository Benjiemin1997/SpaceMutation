import unittest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.UNet.model_unet import UNet
from mut.fgsm_fuzz import fgsm_fuzz_weight, fgsm_fuzz_weight_mnist


class TestFGSMFuzzWeight(unittest.TestCase):
    def setUp(self):
        self.model = UNet()
        self.data_loader = DataLoader(torch.randn(10, 1, 28, 28), batch_size=10, shuffle=True)
        self.epsilon = 0.1

    def test_fgsm_fuzz_weight(self):
        mutated_model = fgsm_fuzz_weight(self.model, self.data_loader, self.epsilon)
        self.assertIsNotNone(mutated_model)

    def test_fgsm_fuzz_weight_mnist(self):
        mutated_model = fgsm_fuzz_weight_mnist(self.model, self.data_loader, self.epsilon)
        self.assertIsNotNone(mutated_model)

    def test_assertions(self):
        # Check for correct assertions in your functions
        # This part is highly dependent on your implementation details
        pass

if __name__ == '__main__':
    unittest.main()