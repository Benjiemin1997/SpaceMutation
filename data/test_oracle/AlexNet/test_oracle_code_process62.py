import unittest
from unittest.mock import patch, Mock

import torch

from models.AlexNet.model_alexnet import AlexNet
from mut.fgsm_fuzz import fgsm_fuzz_weight


class TestFGSMFuzzWeight(unittest.TestCase):

    @patch('torch.cuda.is_available', return_value=True)
    def setUp(self, mock_cuda_available):
        self.device = 'cuda'
        self.model = AlexNet().to(self.device)
        self.data_loader = Mock()
        self.epsilon = 0.1

    def test_fgsm_fuzz_weight(self):
        # Create a mock data_loader and labels
        images = torch.randn(1, 3, 224, 224).to(self.device)
        labels = torch.tensor([0]).to(self.device)
        self.data_loader.__iter__.return_value = [(images, labels)]

        # Call the function under test
        mutated_model = fgsm_fuzz_weight(self.model, self.data_loader, self.epsilon)

        # Assertions to verify the output model is mutated
        self.assertNotEqual(id(mutated_model.features[0].weight), id(self.model.features[0].weight))
        self.assertNotEqual(id(mutated_model.classifier[0].weight), id(self.model.classifier[0].weight))

        # Check if the model's parameters have been updated
        for param in mutated_model.parameters():
            self.assertTrue(param.grad is not None)

        # Check if the model has been zero_grad after calculating gradients
        mutated_model.zero_grad.assert_called_once()

        # Check if the model's state_dict has been updated
        self.assertNotEqual(mutated_model.state_dict(), self.model.state_dict())

if __name__ == '__main__':
    unittest.main()
