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
        # This is a high-level check, as exact values will vary due to randomness in the mutation process.
        original_params = {name: param.clone().detach().requires_grad_(True) for name, param in self.model.named_parameters()}
        mutated_params = {name: param.clone().detach().requires_grad_(True) for name, param in mutated_model.named_parameters()}

        # Assert that at least one parameter has changed
        changed = False
        for name in original_params.keys():
            if not torch.equal(original_params[name], mutated_params[name]):
                changed = True
                break
        self.assertTrue(changed)

        # Clean up
        del mutated_model
        del original_params
        del mutated_params

if __name__ == '__main__':
    unittest.main()

