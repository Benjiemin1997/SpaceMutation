import unittest
from unittest.mock import patch, MagicMock
from mut.replace_activation import replace_activations
from torchvision.models import vgg16
import torch

class TestReplaceActivations(unittest.TestCase):

    @patch('torch.nn.Module')
    def test_replace_activations(self, mock_module):

        mock_model = MagicMock(spec=vgg16(weights='IMAGENET1K_V1'))

        replace_activations(mock_model)


        self.assertEqual(mock_model.features[3].__class__, torch.nn.ELU)
        self.assertEqual(mock_model.features[10].__class__, torch.nn.PReLU)
        self.assertEqual(mock_model.features[17].__class__, torch.nn.SELU)
        self.assertEqual(mock_model.features[24].__class__, torch.nn.SELU)
        self.assertEqual(mock_model.features[29].__class__, torch.nn.ELU)
        self.assertEqual(mock_model.classifier[1].__class__, torch.nn.SELU)
        self.assertEqual(mock_model.classifier[3][0].__class__, torch.nn.Tanh)

if __name__ == '__main__':
    unittest.main()
