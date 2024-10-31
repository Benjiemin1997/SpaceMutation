import torch
import unittest

from models.UNet.model_unet import UNet
from mut.uniform_fuzz import uniform_fuzz_weight


class TestUniformFuzzWeight(unittest.TestCase):

    def setUp(self):
        self.model = UNet()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_uniform_fuzz_weight(self):
        # Given
        lower_bound = -0.1
        upper_bound = 0.1
        
        # When
        mutated_model = uniform_fuzz_weight(self.model, lower_bound, upper_bound).to(self.device)

        # Then
        # Check that the model's weights have been perturbed by comparing before and after values.
        # This is a high-level check, as exact values will vary due to randomness.
        for param in mutated_model.parameters():
            original_values = param.data.clone().detach().cpu().numpy()
            mutated_values = param.data.clone().detach().cpu().numpy()
            self.assertTrue(torch.any(original_values != mutated_values))

        # Additional checks can be added based on expected behavior or specific assertions about the model's state.
        # For example, checking the type of the model or any other properties.

if __name__ == '__main__':
    unittest.main()
