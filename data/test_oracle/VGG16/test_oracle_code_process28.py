import unittest
from unittest.mock import patch, MagicMock
from mut.replace_activation import replace_activations
from torchvision.models import vgg16
import torch

class TestReplaceActivations(unittest.TestCase):

    @patch('torch.nn.Module')
    def test_replace_activations(self, mock_module):
        # Create a mock model
        mock_model = MagicMock(spec=vgg16(weights='IMAGENET1K_V1'))
        # Replace some activation functions in the model
        replace_activations(mock_model)

        # Check that the correct number of activation replacements were made
        self.assertEqual(mock_model.features[3].__class__, torch.nn.ELU)
        self.assertEqual(mock_model.features[10].__class__, torch.nn.PReLU)
        self.assertEqual(mock_model.features[17].__class__, torch.nn.SELU)
        self.assertEqual(mock_model.features[24].__class__, torch.nn.SELU)
        self.assertEqual(mock_model.features[29].__class__, torch.nn.ELU)
        self.assertEqual(mock_model.classifier[0].__class__, torch.nn.Linear)
        self.assertEqual(mock_model.classifier[3][0].__class__, torch.nn.Linear)

        # Check that the replacements are of the correct class
        self.assertTrue(isinstance(mock_model.features[3], torch.nn.ELU))
        self.assertTrue(isinstance(mock_model.features[10], torch.nn.PReLU))
        self.assertTrue(isinstance(mock_model.features[17], torch.nn.SELU))
        self.assertTrue(isinstance(mock_model.features[24], torch.nn.SELU))
        self.assertTrue(isinstance(mock_model.features[29], torch.nn.ELU))
        self.assertTrue(isinstance(mock_model.classifier[0], torch.nn.Linear))
        self.assertTrue(isinstance(mock_model.classifier[3][0], torch.nn.Linear))

if __name__ == '__main__':
    unittest.main()
