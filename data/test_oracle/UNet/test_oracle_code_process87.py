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

        # Original model evaluation
        original_outputs = self.model(data_loader)
        original_loss = torch.nn.functional.cross_entropy(original_outputs, torch.tensor([0]))
        print(f"Original Loss: {original_loss.item()}")

        # Apply FGSM fuzzing
        mutated_model = fgsm_fuzz_weight(self.model, data_loader, epsilon)

        # Model evaluation after FGSM fuzzing
        mutated_outputs = mutated_model(data_loader)
        mutated_loss = torch.nn.functional.cross_entropy(mutated_outputs, torch.tensor([0]))
        print(f"Mutated Loss: {mutated_loss.item()}")

        # Check if loss has changed
        self.assertNotEqual(original_loss.item(), mutated_loss.item())

if __name__ == '__main__':
    unittest.main()



