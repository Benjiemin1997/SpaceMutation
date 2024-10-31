import torch
import unittest

from mut.uniform_fuzz import uniform_fuzz_weight


class TestUniformFuzzWeight(unittest.TestCase):

    def setUp(self):
        self.model = torch.nn.Sequential(
            torch.nn.Conv2d(1, 6, kernel_size=(5, 5), stride=(1, 1)),
            torch.nn.GELU(approximate='none'),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(6, 16, kernel_size=(5, 5), stride=(1, 1)),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Flatten(),
            torch.nn.Linear(in_features=256, out_features=120),
            torch.nn.Sequential(
                torch.nn.Linear(in_features=120, out_features=84),
                torch.nn.PReLU(num_parameters=1)
            ),
            torch.nn.Linear(in_features=84, out_features=10)
        )

    def test_uniform_fuzz_weight(self):
        # Test with provided bounds
        expected_model = uniform_fuzz_weight(self.model, lower_bound=-0.1, upper_bound=0.1)
        # Check that all weights have been modified within the given bounds
        for param in expected_model.parameters():
            self.assertTrue(torch.allclose(param.data, torch.rand_like(param.data) * (0.1 - (-0.1)) - 0.1, atol=1e-4))

        # Test with default bounds
        expected_model_default = uniform_fuzz_weight(self.model)
        # Check that all weights have been modified within the default bounds
        for param in expected_model_default.parameters():
            self.assertTrue(torch.allclose(param.data, torch.rand_like(param.data) * (0.01 - (-0.01)) - 0.01, atol=1e-4))

if __name__ == '__main__':
    unittest.main()
