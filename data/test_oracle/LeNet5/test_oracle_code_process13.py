import unittest
import torch
from torch.utils.data import DataLoader

from models.LeNet5.model_lenet5 import LeNet5
from mut.fgsm_fuzz import fgsm_fuzz_weight


class TestFgsmFuzzWeight(unittest.TestCase):

    def setUp(self):
        self.model = LeNet5()
        self.data_loader = DataLoader(torch.randn(10, 1, 28, 28), batch_size=10)
        self.epsilon = 0.1

    def test_fgsm_fuzz_weight(self):
        mutated_model = fgsm_fuzz_weight(self.model, self.data_loader, self.epsilon)
        self.assertIsNotNone(mutated_model)

    def test_fgsm_fuzz_weight_mnist(self):
        mutated_model = fgsm_fuzz_weight(self.model, self.data_loader, self.epsilon)
        self.assertIsNotNone(mutated_model)

    def test_assertions(self):
        # Check for correct assertions in your functions
        pass  # Implement specific checks here

if __name__ == '__main__':
    unittest.main()
