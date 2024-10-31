import unittest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.AlexNet.model_alexnet import AlexNet
from mut.fgsm_fuzz import fgsm_fuzz_weight


class TestFGM(unittest.TestCase):

    def setUp(self):
        self.model = AlexNet()
        self.data_loader = DataLoader(torch.randn(1, 3, 224, 224), batch_size=1)
        self.epsilon = 0.1
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_fgsm_fuzz_weight(self):
        self.model.to(self.device)
        original_model_state = self.model.state_dict()

        # Apply FGSM fuzzing
        fgsm_fuzz_weight(self.model, self.data_loader, self.epsilon)

        # Check that model parameters have been altered
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                self.assertFalse(torch.equal(param.data, original_model_state[name]), msg=f"Parameter {name} has not been altered")

if __name__ == '__main__':
    unittest.main()
