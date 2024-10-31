import torch
import unittest
from torch import nn

from mut.guassian_fuzz_splayers import gaussian_fuzzing_splayer
from mut.random_shuffle import random_shuffle_weight
from mut.remove_activation import remove_activations
from mut.replace_activation import replace_activations
from mut.uniform_fuzz import uniform_fuzz_weight

class TestModelMutations(unittest.TestCase):

    def setUp(self):
        # Create a simple model for testing
        class SimpleModel(nn.Module):
            def __init__(self):
                super(SimpleModel, self).__init__()
                self.linear = nn.Linear(10, 10)
                self.relu = nn.ReLU()

            def forward(self, x):
                return self.relu(self.linear(x))

        self.model = SimpleModel()

    def test_gaussian_fuzzing_splayer(self):
        # Apply Gaussian Fuzzing to the model
        mutated_model = gaussian_fuzzing_splayer(self.model)

        # Check if the mutated model has changed
        self.assertFalse(torch.allclose(self.model.linear.weight, mutated_model.linear.weight),
                         "Gaussian fuzzing did not change the model weights.")

    def test_random_shuffle_weight(self):
        # Apply Random Shuffle to the model's weight
        mutated_model = random_shuffle_weight(self.model)

        # Check if the mutated model has changed
        self.assertFalse(torch.equal(self.model.linear.weight, mutated_model.linear.weight),
                         "Random shuffle did not change the model weights.")

    def test_remove_activations(self):
        # Apply Remove Activations to the model
        mutated_model = remove_activations(self.model)

        # Check if the mutated model has removed activations
        self.assertNotIn('relu', str(mutated_model), "ReLU was not removed from the model.")

    def test_replace_activations(self):
        # Apply Replace Activations to the model
        # Assuming replace_activations accepts a model and a new activation function
        mutated_model = replace_activations(self.model, nn.Sigmoid())

        # Check if the mutated model has replaced activations
        self.assertNotIn('relu', str(mutated_model), "ReLU was not removed from the model.")
        self.assertIn('Sigmoid', str(mutated_model), "Sigmoid was not added to the model.")

    def test_uniform_fuzz_weight(self):
        # Apply Uniform Fuzzing to the model's weight
        mutated_model = uniform_fuzz_weight(self.model)

        # Check if the mutated model has changed
        self.assertFalse(torch.allclose(self.model.linear.weight, mutated_model.linear.weight),
                         "Uniform fuzzing did not change the model weights.")

if __name__ == '__main__':
    unittest.main()