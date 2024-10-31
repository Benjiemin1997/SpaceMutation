import unittest
from unittest.mock import patch, Mock

import torch
from torch import nn

from models.AlexNet.model_alexnet import AlexNet


class TestFGSMFuzzWeight(unittest.TestCase):

    @patch('torch.cuda.is_available', return_value=True)
    def setUp(self, _):
        self.device = 'cuda'
        self.model = AlexNet()
        self.data_loader = Mock()
        self.epsilon = 0.1

    def test_fgsm_fuzz_weight(self):
        self.model.eval()
        self.model.to(self.device)
        self.model.zero_grad()

        # Fuzzing the weights using FGSM
        fuzzed_model = fgsm_fuzz_weight(self.model, self.data_loader, self.epsilon)

        # Assertions to check if the model's parameters have been fuzzed correctly
        for param in fuzzed_model.parameters():
            if param.requires_grad:
                self.assertTrue(torch.any(param.grad.sign() != 0),
                                "Grad sign should not be zero after FGSM fuzzing")

if __name__ == '__main__':
    unittest.main()
