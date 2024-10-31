import torch
import unittest

from models.LeNet5.model_lenet5 import LeNet5
from mut.uniform_fuzz import uniform_fuzz_weight


class TestUniformFuzzWeight(unittest.TestCase):

    def setUp(self):
        self.model = LeNet5()
        self.lower_bound = -0.1
        self.upper_bound = 0.1
        
    def test_model_uniform_fuzz(self):
        # Apply uniform fuzzing to model weights
        self.model = uniform_fuzz_weight(self.model, self.lower_bound, self.upper_bound)

        # Test assertion: Check that all parameters have been modified by the uniform fuzzing process
        for param in self.model.parameters():
            if param.requires_grad:
                # Use torch.allclose to check if the difference between the original and modified parameters is within a certain threshold
                diff = torch.abs(param - param.data.clone().detach())
                self.assertTrue(torch.allclose(diff, torch.zeros_like(diff), atol=0.01, rtol=0.01))

if __name__ == '__main__':
    unittest.main()
