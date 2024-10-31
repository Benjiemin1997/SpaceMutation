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
        original_params = {name: param.clone().detach().requires_grad_(True) for name, param in self.model.named_parameters()}
        mutated_params = {name: param.clone().detach().requires_grad_(True) for name, param in mutated_model.named_parameters()}

        any_param_changed = any(not torch.allclose(original_params[name], mutated_params[name]) for name in original_params.keys())
        self.assertTrue(any_param_changed, "No parameters were mutated as expected.")

if __name__ == '__main__':
    unittest.main()

