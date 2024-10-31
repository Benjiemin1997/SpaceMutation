import torch
import unittest

from mut.uniform_fuzz import uniform_fuzz_weight


class TestUniformFuzzWeight(unittest.TestCase):

    def setUp(self):
        self.model = AlexNet()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_uniform_fuzz_weight(self):
        # Test case for the uniform fuzz weight function
        original_state_dict = self.model.state_dict()

        # Apply the uniform fuzzing to the model's weights
        uniform_fuzz_weight(self.model)

        # Assert that the state dict has changed
        self.assertNotEqual(original_state_dict, self.model.state_dict())

        # Assert that all parameters have been perturbed by a uniform noise
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                noise = torch.rand(param.size()).to(param.device) * (0.1 - (-0.1)) + (-0.1)
                self.assertTrue(torch.allclose(param.data - noise, self.model.state_dict()[name], atol=1e-3))

if __name__ == '__main__':
    unittest.main()
