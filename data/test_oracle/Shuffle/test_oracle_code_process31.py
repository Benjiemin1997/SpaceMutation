import torch
import unittest

from models.Shufflenetv2.shufflenetv2 import ShuffleNetV2


class TestUniformFuzzWeight(unittest.TestCase):

    def setUp(self):
        self.model = ShuffleNetV2().eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def tearDown(self):
        del self.model

    def test_uniform_fuzz_weight(self):
        # Given
        lower_bound = -0.1
        upper_bound = 0.1
        expected_weight_changes = {}  # Placeholder for expected weight changes

        # When
        mutated_model = uniform_fuzz_weight(self.model, lower_bound, upper_bound)

        # Then
        # Assert that all model parameters have been perturbed within the given bounds
        for name, param in mutated_model.named_parameters():
            if param.requires_grad:
                perturbation = param.data - torch.rand_like(param.data) * (upper_bound - lower_bound) - lower_bound
                expected_weight_changes[name] = perturbation.tolist()

        # Check if the perturbation is within the expected bounds
        for name, param in mutated_model.named_parameters():
            if param.requires_grad:
                perturbation = param.data - torch.rand_like(param.data) * (upper_bound - lower_bound) - lower_bound
                self.assertTrue(torch.all(perturbation >= lower_bound))
                self.assertTrue(torch.all(perturbation <= upper_bound))

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)