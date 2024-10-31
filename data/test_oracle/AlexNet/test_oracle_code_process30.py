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
        
        # Call the function to replace activations
        mutated_model = replace_activations(mock_model)
        
        # Check if the model has been mutated by verifying the types of modules
        self.assertNotEqual(type(mutated_model[0]), nn.ReLU)
        self.assertNotEqual(type(mutated_model[1]), nn.LeakyReLU)
        self.assertNotEqual(type(mutated_model[2]), nn.Sigmoid)
        self.assertNotEqual(type(mutated_model[3]), nn.Tanh)
        self.assertNotEqual(type(mutated_model[4]), nn.ELU)
        self.assertNotEqual(type(mutated_model[5]), nn.PReLU)
        self.assertNotEqual(type(mutated_model[6]), nn.SELU)
        self.assertNotEqual(type(mutated_model[7]), nn.GELU)
        self.assertEqual(len(mutated_model), 8)
        

        pass
        
if __name__ == '__main__':
    unittest.main()
