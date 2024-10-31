import unittest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.LeNet5.model_lenet5 import LeNet5
from mut.fgsm_fuzz import fgsm_fuzz_weight, fgsm_fuzz_weight_mnist


class TestFGSMFuzz(unittest.TestCase):

    def setUp(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = LeNet5().to(self.device)
        self.data_loader = DataLoader(torch.randn(10, 1, 28, 28).to(self.device), batch_size=1)
        self.epsilon = 0.1

    def test_fgsm_fuzz_weight(self):
        self.model.eval()
        fuzzed_model = fgsm_fuzz_weight(self.model, self.data_loader, self.epsilon)
        self.assertIsNotNone(fuzzed_model)

    def test_fgsm_fuzz_weight_mnist(self):
        self.model.eval()
        fuzzed_model = fgsm_fuzz_weight_mnist(self.model, self.data_loader, self.epsilon)
        self.assertIsNotNone(fuzzed_model)

if __name__ == '__main__':
    unittest.main()
