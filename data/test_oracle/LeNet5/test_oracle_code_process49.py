import torch
from torch import nn
from mut.guassian_fuzz_splayers import gaussian_fuzzing_splayer
from mut.random_shuffle import random_shuffle_weight
from mut.remove_activation import remove_activations
from mut.replace_activation import replace_activations

from mut.uniform_fuzz import uniform_fuzz_weight

# Define a simple test model
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.linear1 = nn.Linear(10, 5)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        return x

# Test cases for different mutation operations
def test_gaussian_fuzzing():
    model = SimpleModel()
    mutated_model = gaussian_fuzzing_splayer(model)

    assert any(torch.norm(param.grad) > 0 for param in mutated_model.parameters())

def test_random_shuffle():
    model = SimpleModel()
    mutated_model = random_shuffle_weight(model)

    assert not all(torch.equal(mutated_model.linear1.weight, model.linear1.weight))

def test_remove_activations():
    model = SimpleModel()
    mutated_model = remove_activations(model)

    assert not hasattr(mutated_model.linear1, 'next_module')

def test_replace_activations():
    model = SimpleModel()
    mutated_model = replace_activations(model)
    # For example: Check if the linear layer now uses a different activation function
    assert isinstance(mutated_model.linear1.next_module, nn.Sigmoid)

def test_uniform_fuzz():
    model = SimpleModel()
    mutated_model = uniform_fuzz_weight(model)
    # For example: Check if any of the parameters have been uniformly fuzzed
    assert any(torch.norm(param.grad) > 0 for param in mutated_model.parameters())