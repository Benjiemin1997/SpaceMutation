import unittest
from unittest.mock import patch, Mock
from torch import nn

from mut.replace_activation import replace_activations


class TestReplaceActivations(unittest.TestCase):

    @patch('torch.nn.modules.activation.ReLU', new=Mock)
    @patch('torch.nn.modules.activation.LeakyReLU', new=Mock)
    @patch('torch.nn.modules.activation.Sigmoid', new=Mock)
    @patch('torch.nn.modules.activation.Tanh', new=Mock)
    @patch('torch.nn.modules.activation.ELU', new=Mock)
    @patch('torch.nn.modules.activation.PReLU', new=Mock)
    @patch('torch.nn.modules.activation.SELU', new=Mock)
    @patch('torch.nn.modules.activation.GELU', new=Mock)
    
    def test_replace_activations(self):
        # Create a mock model with specific activation functions
        mock_model = nn.Sequential(
            nn.ReLU(),
            nn.LeakyReLU(),
            nn.Sigmoid(),
            nn.Tanh(),
            nn.ELU(),
            nn.PReLU(),
            nn.SELU(),
            nn.GELU()
        )
        
        # Call the function under test
        modified_model = replace_activations(mock_model)
        
        # Check that the number of layers has changed
        self.assertNotEqual(len(modified_model), len(mock_model))
        
        # Check that some activation functions have been replaced
        for i in range(len(modified_model)):
            layer = modified_model[i]
            if isinstance(layer, nn.ReLU) or isinstance(layer, nn.LeakyReLU) or \
               isinstance(layer, nn.Sigmoid) or isinstance(layer, nn.Tanh) or \
               isinstance(layer, nn.ELU) or isinstance(layer, nn.PReLU) or \
               isinstance(layer, nn.SELU) or isinstance(layer, nn.GELU):
                self.assertNotEqual(type(layer), type(mock_model[i]))
                
if __name__ == '__main__':
    unittest.main()
