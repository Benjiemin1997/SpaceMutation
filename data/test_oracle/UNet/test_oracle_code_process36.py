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
            

            self.assertTrue(torch.any(perturbed_param != original_param))


            self.assertTrue(torch.all(lower_bound <= perturbed_param))
            self.assertTrue(torch.all(perturbed_param <= upper_bound))

if __name__ == '__main__':
    unittest.main()
