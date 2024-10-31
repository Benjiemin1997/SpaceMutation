import torch
from torch import nn
from mut.guassian_fuzz_splayers import gaussian_fuzzing_splayer
from mut.random_shuffle import random_shuffle_weight
from mut.remove_activation import remove_activations
from mut.replace_activation import replace_activations
from mut.uniform_fuzz import uniform_fuzz_weight

class TestModelMutations:

    def test_gaussian_fuzzing_splayer(self):
        # Create a simple model for testing
        class SimpleModel(nn.Module):
            def __init__(self):
                super(SimpleModel, self).__init__()
                self.linear = nn.Linear(10, 10)

            def forward(self, x):
                return self.linear(x)

        model = SimpleModel()

        # Apply Gaussian Fuzzing to the model
        mutated_model = gaussian_fuzzing_splayer(model)

        # Check if the mutated model has changed
        assert not torch.allclose(model.linear.weight, mutated_model.linear.weight)

    def test_random_shuffle_weight(self):
        # Create a simple model for testing
        class SimpleModel(nn.Module):
            def __init__(self):
                super(SimpleModel, self).__init__()
                self.linear = nn.Linear(10, 10)

            def forward(self, x):
                return self.linear(x)

        model = SimpleModel()

        # Apply Random Shuffle to the model's weight
        mutated_model = random_shuffle_weight(model)

        # Check if the mutated model has changed
        assert not torch.equal(model.linear.weight, mutated_model.linear.weight)

    def test_remove_activations(self):
        # Create a simple model with activation layers for testing
        class SimpleModel(nn.Module):
            def __init__(self):
                super(SimpleModel, self).__init__()
                self.linear1 = nn.Linear(10, 10)
                self.relu = nn.ReLU()

            def forward(self, x):
                return self.relu(self.linear1(x))

        model = SimpleModel()


        mutated_model = remove_activations(model)

        # Check if the mutated model has removed activations
        assert 'relu' not in str(mutated_model)

    def test_replace_activations(self):
        # Create a simple model with activation layers for testing
        class SimpleModel(nn.Module):
            def __init__(self):
                super(SimpleModel, self).__init__()
                self.linear1 = nn.Linear(10, 10)
                self.relu = nn.ReLU()

            def forward(self, x):
                return self.relu(self.linear1(x))

        model = SimpleModel()

        # Apply Replace Activations to the model
        mutated_model = replace_activations(model)

        # Check if the mutated model has replaced activations
        assert 'relu' not in str(mutated_model)
        assert 'sigmoid' in str(mutated_model)

    def test_uniform_fuzz_weight(self):
        # Create a simple model for testing
        class SimpleModel(nn.Module):
            def __init__(self):
                super(SimpleModel, self).__init__()
                self.linear = nn.Linear(10, 10)

            def forward(self, x):
                return self.linear(x)

        model = SimpleModel()

        # Apply Uniform Fuzzing to the model's weight
        mutated_model = uniform_fuzz_weight(model)

        # Check if the mutated model has changed
        assert not torch.allclose(model.linear.weight, mutated_model.linear.weight)
