import torch
import unittest

from models.Shufflenetv2.shufflenetv2 import ShuffleNetV2
from mut.uniform_fuzz import uniform_fuzz_weight


class TestUniformFuzzWeight(unittest.TestCase):

    def setUp(self):
        self.model = ShuffleNetV2().eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def tearDown(self):
        del self.model

    def test_uniform_fuzz_weight(self):
        lower_bound = -0.1
        upper_bound = 0.1
        model_copy = self.model.clone().to(self.device)

        # Apply uniform fuzzing to the model
        uniform_fuzz_weight(model=self.model, lower_bound=lower_bound, upper_bound=upper_bound)

        # Check that the original and mutated models have different weights
        for param, copy_param in zip(self.model.parameters(), model_copy.parameters()):
            if param.requires_grad:
                self.assertNotEqual(param.data, copy_param.data, "The models should have different weights after uniform fuzzing.")

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)