import unittest
from unittest.mock import patch

from torch import nn

from models.UNet.model_unet import UNet
from mut.replace_activation import replace_activations

class TestReplaceActivations(unittest.TestCase):

    def setUp(self):
        self.model = UNet()  # Initialize your model here


    @patch('torch.nn.functional.relu')  # Mock the torch.nn.functional.relu to simulate its behavior during the test
    @patch('random.choice', side_effect=[nn.LeakyReLU(), nn.Sigmoid(), nn.Tanh(), nn.ELU(), nn.PReLU(), nn.SELU(), nn.GELU()])
    def test_replace_activations(self, mock_choice, mock_relu):
        replace_activations(self.model)
        
        # Assert that all ReLU activations have been replaced
        for name, module in self.model.named_modules():
            if isinstance(module, nn.ReLU):
                self.assertNotIsInstance(module, nn.ReLU)
                
        # Compare the modified model's state_dict with the expected one
        self.assertDictEqual(self.model.state_dict(), self.expected_model.state_dict())

if __name__ == '__main__':
    unittest.main()
