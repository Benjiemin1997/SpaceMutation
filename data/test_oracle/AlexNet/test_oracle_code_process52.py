import unittest
from unittest.mock import patch, Mock

import torch
from torch import Tensor
from torch.nn import CrossEntropyLoss

from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset

from models.AlexNet.model_alexnet import AlexNet
from mut.fgsm_fuzz import fgsm_fuzz_weight


class TestFgsmFuzzWeight(unittest.TestCase):
    def setUp(self):
        self.model = AlexNet()
        self.data_loader = DataLoader(Dataset(), batch_size=1)
        self.epsilon = 0.1
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.loss_function = CrossEntropyLoss()

    @patch('torch.nn.Module.zero_grad')
    @patch('torch.autograd.set_detect_anomaly')
    @patch('torch.nn.Module.state_dict')
    def test_fgsm_fuzz_weight(self, mock_state_dict, mock_set_detect_anomaly, mock_zero_grad):
        mock_state_dict.return_value = {'weight': Tensor([1.0])}
        mock_zero_grad.return_value = None
        mock_set_detect_anomaly.return_value = None

        # Apply FGSM fuzzing
        self.model = fgsm_fuzz_weight(self.model, self.data_loader, self.epsilon)


        self.assertIsNotNone(self.model)
        self.assertTrue(hasattr(self.model, 'parameters'))
        self.assertTrue(hasattr(self.model.parameters(), '__iter__'))

if __name__ == '__main__':
    unittest.main()
