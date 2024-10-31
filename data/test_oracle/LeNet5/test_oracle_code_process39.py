import unittest
from unittest.mock import patch, MagicMock

from torch import nn

from models.LeNet5.model_lenet5 import LeNet5
from mut.replace_activation import replace_activations


class TestReplaceActivations(unittest.TestCase):

    @patch('your_module.random.choice', side_effect=[MagicMock(), MagicMock()])
    @patch('your_module.random.shuffle', side_effect=[MagicMock(), MagicMock()])
    @patch('your_module.uniform_fuzz_weight', side_effect=[MagicMock(), MagicMock()])
    @patch('your_module.gaussian_fuzzing_splayer', side_effect=[MagicMock(), MagicMock()])
    @patch('your_module.remove_activations', side_effect=[MagicMock(), MagicMock()])
    def test_replace_activations(self, mock_remove_activations, mock_gaussian_fuzzing_splayer, 
                                 mock_uniform_fuzz_weight, mock_random_shuffle_weight, mock_random_choice):
        # Define the initial model structure as per your example
        initial_model = LeNet5()  # Replace with actual model definition
        
        # Call the replace_activations function on the initial model
        mutated_model = replace_activations(initial_model)
        

        self.assertTrue(isinstance(mutated_model, nn.Module))
        self.assertEqual(len(list(mutated_model.named_modules())), len(initial_model.named_modules()) + 1)  # Assuming one more module due to the added activation replacement
        

        mock_remove_activations.assert_called_once()
        mock_gaussian_fuzzing_splayer.assert_called_once()
        mock_uniform_fuzz_weight.assert_called_once()
        mock_random_shuffle_weight.assert_called_once()
        mock_random_choice.assert_called_with([nn.ReLU(), nn.LeakyReLU(), nn.Sigmoid(), nn.Tanh(), nn.ELU(), nn.PReLU(), nn.SELU(), nn.GELU()])

if __name__ == '__main__':
    unittest.main()
