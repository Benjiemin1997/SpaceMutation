import unittest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.UNet.model_unet import UNet
from mut.fgsm_fuzz import fgsm_fuzz_weight, fgsm_fuzz_weight_mnist


class TestFGSMFuzzWeight(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = UNet().to(self.device)
        self.data_loader = DataLoader(range(10), batch_size=5, shuffle=True)
        self.epsilon = 0.1

    def test_fgsm_fuzz_weight(self):
        original_model = self.model.state_dict().copy()
        modified_model = fgsm_fuzz_weight(self.model, self.data_loader, self.epsilon)
        
        for param_name, original_param in original_model.items():
            modified_param = modified_model[param_name]
            # Check that parameters have been perturbed by epsilon
            self.assertTrue(torch.allclose(original_param, modified_param, atol=self.epsilon))

    def test_fgsm_fuzz_weight_mnist(self):
        original_model = self.model.state_dict().copy()
        modified_model = fgsm_fuzz_weight_mnist(self.model, self.data_loader, self.epsilon)
        
        for param_name, original_param in original_model.items():
            modified_param = modified_model[param_name]
            # Check that parameters have been perturbed by epsilon
            self.assertTrue(torch.allclose(original_param, modified_param, atol=self.epsilon))

if __name__ == '__main__':
    unittest.main()
