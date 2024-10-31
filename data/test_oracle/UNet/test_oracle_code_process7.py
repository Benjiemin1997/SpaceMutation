import unittest
import torch
import numpy as np

from models.UNet.model_unet import UNet
from mut.fgsm_fuzz import fgsm_fuzz_weight


class TestFGSMFuzzWeight(unittest.TestCase):
    def setUp(self):
        self.model = UNet()

    def test_fgsm_fuzz_weight(self):
        epsilon = 0.1
        data_loader = torch.utils.data.DataLoader(torch.randn(1, 1, 28, 28), batch_size=1)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Original model state
        original_state = {}
        for name, param in self.model.named_parameters():
            original_state[name] = param.clone().detach().requires_grad_(True).to(device)

        mutated_model = fgsm_fuzz_weight(self.model, data_loader, epsilon)


        for name, param in self.model.named_parameters():
            assert not torch.equal(param, original_state[name]), f"Parameter {name} is not mutated."
        for name, param in self.model.named_parameters():
            param.data = original_state[name]

if __name__ == '__main__':
    unittest.main()
