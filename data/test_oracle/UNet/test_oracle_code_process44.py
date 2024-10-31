import unittest
import torch
import numpy as np

from models.UNet.model_unet import UNet
from mut.fgsm_fuzz import fgsm_fuzz_weight, fgsm_fuzz_weight_mnist


class TestFGSMFuzzWeight(unittest.TestCase):
    def setUp(self):
        self.model = UNet()
        self.data_loader = torch.utils.data.DataLoader(torch.randn(1, 1, 32, 32), batch_size=1)
        self.epsilon = 0.1

    def test_fgsm_fuzz_weight(self):
        model = fgsm_fuzz_weight(self.model, self.data_loader, self.epsilon)
        # Add assertions to check the model's weights have been perturbed
        with torch.no_grad():
            for param in model.parameters():
                if param.requires_grad:
                    assert not torch.allclose(param, param.clone().detach()), "Weights have not been perturbed."
                    break

    def test_fgsm_fuzz_weight_mnist(self):
        model = fgsm_fuzz_weight_mnist(self.model, self.data_loader, self.epsilon)
        # Add assertions to check the model's weights have been perturbed
        with torch.no_grad():
            for param in model.parameters():
                if param.requires_grad:
                    assert not torch.allclose(param, param.clone().detach()), "Weights have not been perturbed."
                    break

if __name__ == '__main__':
    unittest.main()
