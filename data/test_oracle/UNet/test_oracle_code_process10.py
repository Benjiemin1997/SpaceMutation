import unittest
import torch
from torch.utils.data import DataLoader

from models.UNet.model_unet import UNet
from mut.fgsm_fuzz import fgsm_fuzz_weight


class TestFGSMFuzzWeight(unittest.TestCase):

    def setUp(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = UNet().to(self.device)
        self.data_loader = DataLoader(torch.randn(10, 1, 28, 28), batch_size=10, shuffle=False)
        self.epsilon = 0.1

    def test_fgsm_fuzz_weight(self):
        mutated_model = fgsm_fuzz_weight(self.model, self.data_loader, self.epsilon)
        original_weights = [param.clone().detach().requires_grad_(True) for param in mutated_model.parameters()]
        mutated_weights = [param.detach().requires_grad_(True) for param in mutated_model.parameters()]
        for original, mutated in zip(original_weights, mutated_weights):
            self.assertFalse(torch.equal(original, mutated), "Weights have not been mutated.")

    def test_fgsm_fuzz_weight_mnist(self):
        mutated_model = fgsm_fuzz_weight(self.model, self.data_loader, self.epsilon)

        original_output_weights = [param.clone().detach().requires_grad_(True) for param in mutated_model.final_conv[0].parameters()]
        mutated_output_weights = [param.detach().requires_grad_(True) for param in mutated_model.final_conv[0].parameters()]
        for original, mutated in zip(original_output_weights, mutated_output_weights):
            self.assertFalse(torch.equal(original, mutated), "Output layer weights have not been mutated.")

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
