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
        # Given
        lower_bound = -0.1
        upper_bound = 0.1
        expected_weight_changes = {}  # Placeholder for expected weight changes

        # When
        mutated_model = uniform_fuzz_weight(self.model, lower_bound, upper_bound)

        # Then
        for param in mutated_model.parameters():
            if param.requires_grad:
                weight_changes = param.data - self.model.state_dict()[param.name]
                self.assertTrue((weight_changes >= lower_bound).all())
                self.assertTrue((weight_changes <= upper_bound).all())
                expected_weight_changes[param.name] = weight_changes

        # Check that the expected changes match the actual changes
        for name, expected_change in expected_weight_changes.items():
            actual_change = mutated_model.state_dict()[name] - self.model.state_dict()[name]
            self.assertTrue(torch.allclose(actual_change, expected_change))

if __name__ == '__main__':
    unittest.main()