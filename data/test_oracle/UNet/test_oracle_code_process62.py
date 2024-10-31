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


        for param in mutated_model.parameters():
            perturbed_param = param.data
            original_param = param.clone().data
            self.assertTrue(torch.allclose(perturbed_param, original_param + torch.rand(original_param.size()) * (upper_bound - lower_bound) + lower_bound, atol=1e-3))


        input_data = torch.randn(1, 1, 32, 32).to(self.device)
        output_before = self.model(input_data)
        _ = mutated_model(input_data)


        output_after = mutated_model(input_data)
        self.assertFalse(torch.allclose(output_before, output_after, atol=1e-3))

if __name__ == '__main__':
    unittest.main()
