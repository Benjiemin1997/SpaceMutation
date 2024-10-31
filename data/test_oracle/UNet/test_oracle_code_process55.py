import unittest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.UNet.model_unet import UNet
from mut.fgsm_fuzz import fgsm_fuzz_weight_mnist, fgsm_fuzz_weight


class TestFGSMFuzzWeight(unittest.TestCase):

    def setUp(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = UNet().to(self.device)
        self.data_loader = DataLoader(torch.randn(10, 1, 28, 28), batch_size=10, shuffle=False)
        self.epsilon = 0.1

    def test_fgsm_fuzz_weight(self):
        mutated_model = fgsm_fuzz_weight(self.model, self.data_loader, self.epsilon)

        self.assertIsNotNone(mutated_model)

        
    def test_fgsm_fuzz_weight_mnist(self):
        mutated_model_mnist = fgsm_fuzz_weight_mnist(self.model, self.data_loader, self.epsilon)

        self.assertIsNotNone(mutated_model_mnist)


if __name__ == '__main__':
    unittest.main()
