import unittest
from unittest.mock import patch, Mock

import torch
from torch import Tensor
from torch.nn import CrossEntropyLoss

from torch.optim import SGD
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset

from models.AlexNet.model_alexnet import AlexNet
from mut.fgsm_fuzz import fgsm_fuzz_weight
from mut.reverse_activation import reverse_activations


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
        mock_state_dict.return_value = {'weight': Tensor([1, 2, 3]), 'bias': Tensor([4, 5, 6])}
        mock_zero_grad.return_value = None
        mock_set_detect_anomaly.return_value = None

        model = reverse_activations(self.model)

        # Assertions based on expected behavior of `fgsm_fuzz_weight` function
        self.assertEqual(len(model.features[0].weight.grad), 9)
        self.assertEqual(len(model.features[0].bias.grad), 9)
        self.assertEqual(len(model.classifier[0].weight.grad), 9216)
        self.assertEqual(len(model.classifier[0].bias.grad), 9216)

        # Ensure that the model's weights have been modified by `epsilon`
        self.assertNotEqual(model.features[0].weight.data.sum().item(), 1)
        self.assertNotEqual(model.features[0].bias.data.sum().item(), 4)
        self.assertNotEqual(model.classifier[0].weight.data.sum().item(), 9216)
        self.assertNotEqual(model.classifier[0].bias.data.sum().item(), 9216)

if __name__ == '__main__':
    unittest.main()
