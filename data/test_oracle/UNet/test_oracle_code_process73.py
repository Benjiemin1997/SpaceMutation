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
        self.data_loader = DataLoader(torch.randn(2, 1, 28, 28).to(self.device), batch_size=2)
        self.epsilon = 0.1

    def test_fgsm_fuzz_weight(self):
        original_model_weights = {param.name: param.clone() for param in self.model.parameters()}
        mutated_model = fgsm_fuzz_weight(self.model, self.data_loader, self.epsilon)
        
        for name, param in mutated_model.named_parameters():
            self.assertNotEqual(param.abs().mean(), original_model_weights[name].abs().mean(), 
                                msg=f"Weights have not been mutated for parameter {name}")

    def test_fgsm_fuzz_weight_mnist(self):
        original_model_weights = {param.name: param.clone() for param in self.model.parameters()}
        mutated_model = fgsm_fuzz_weight_mnist(self.model, self.data_loader, self.epsilon)
        
        for name, param in mutated_model.named_parameters():
            self.assertNotEqual(param.abs().mean(), original_model_weights[name].abs().mean(), 
                                msg=f"Weights have not been mutated for parameter {name}")

if __name__ == '__main__':
    unittest.main()
