import torch
import unittest

from models.Shufflenetv2.shufflenetv2 import ShuffleNetV2
from mut.uniform_fuzz import uniform_fuzz_weight


class TestModelFunction(unittest.TestCase):
    def setUp(self):
        self.model = ShuffleNetV2()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_uniform_fuzz_weight(self):
        # Apply uniform fuzzing to model weights
        uniform_fuzz_weight(self.model, lower_bound=-0.1, upper_bound=0.1)

        # Check that all weights have been perturbed
        for param in self.model.parameters():
            if param.requires_grad:
                self.assertTrue(torch.all(param.data != param.data))  # This check is simplistic and assumes the perturbation is non-zero

    def tearDown(self):
        del self.model

if __name__ == '__main__':
    unittest.main()
