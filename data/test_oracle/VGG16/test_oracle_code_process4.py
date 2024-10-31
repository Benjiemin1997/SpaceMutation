import torch
from torch import nn
from mut.guassian_fuzz_splayers import gaussian_fuzzing_splayer
from mut.random_shuffle import random_shuffle_weight
from mut.remove_activation import remove_activations
from mut.replace_activation import replace_activations
from mut.uniform_fuzz import uniform_fuzz_weight

def test_gaussian_fuzzing_splayer():
    # Initialize a simple model, e.g., a linear layer
    model = nn.Sequential(nn.Linear(10, 10))

    # Apply Gaussian Fuzzing to the model
    model = gaussian_fuzzing_splayer(model)

    # Assert that the model has been mutated
    assert not torch.equal(model[0].weight, torch.randn(10, 10)), "Gaussian fuzzing did not mutate the model"

def test_random_shuffle_weight():
    # Initialize a simple model, e.g., a linear layer with learnable weights
    model = nn.Sequential(nn.Linear(10, 10))

    # Apply Random Shuffle Weight Mutation to the model
    random_shuffle_weight(model)

    # Assert that the model's weights have been shuffled
    assert not torch.equal(model[0].weight, torch.randn(10, 10)), "Random shuffle weight mutation did not mutate the model"

def test_remove_activations():
    # Initialize a simple model with activation layers
    model = nn.Sequential(nn.Linear(10, 10), nn.ReLU())

    # Apply Remove Activations Mutation to the model
    model = remove_activations(model)

    # Assert that the model no longer contains the ReLU activation layer
    assert 'ReLU' not in str(model), "Remove activations mutation did not remove the ReLU layer"

def test_replace_activations():
    # Initialize a simple model with an activation layer
    model = nn.Sequential(nn.Linear(10, 10), nn.ReLU())

    # Replace the ReLU activation with a Sigmoid activation
    model = replace_activations(model, nn.ReLU(), nn.Sigmoid())

    # Assert that the model now contains a Sigmoid activation layer instead of ReLU
    assert 'Sigmoid' in str(model), "Replace activations mutation did not replace the ReLU layer with a Sigmoid layer"

def test_uniform_fuzz_weight():
    # Initialize a simple model, e.g., a linear layer with learnable weights
    model = nn.Sequential(nn.Linear(10, 10))

    # Apply Uniform Fuzz Weight Mutation to the model
    uniform_fuzz_weight(model)

    # Assert that the model's weights have been mutated
    assert not torch.equal(model[0].weight, torch.randn(10, 10)), "Uniform fuzz weight mutation did not mutate the model"