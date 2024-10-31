import torch
from torch import nn
from mut.guassian_fuzz_splayers import gaussian_fuzzing_splayer
from mut.random_shuffle import random_shuffle_weight
from mut.remove_activation import remove_activations
from mut.replace_activation import replace_activations

from mut.uniform_fuzz import uniform_fuzz_weight

# Define a simple test model
class TestModel(nn.Module):
    def __init__(self):
        super(TestModel, self).__init__()
        self.linear1 = nn.Linear(10, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        return x

# Test cases for different mutation operations
def test_gaussian_fuzzing():
    model = TestModel()
    mutated_model = gaussian_fuzzing_splayer(model)
    # Check that at least one weight has been perturbed
    any_perturbed = any(torch.any(param.grad != 0).item() for name, param in mutated_model.named_parameters())
    assert any_perturbed, "No parameters were perturbed by gaussian_fuzzing_splayer."

def test_random_shuffle():
    model = TestModel()
    mutated_model = random_shuffle_weight(model)
    # Check that the order of parameters has changed
    any_shuffled = any(old_name != new_name for old_name, _ in model.named_parameters() 
                       for new_name, _ in mutated_model.named_parameters() 
                       if old_name.startswith(new_name))
    assert any_shuffled, "No parameters were shuffled by random_shuffle_weight."

def test_remove_activations():
    model = TestModel()
    mutated_model = remove_activations(model)
    # Check that the model no longer contains an activation function
    assert 'relu' not in str(mutated_model), "Activation function was not removed by remove_activations."

def test_replace_activations():
    model = TestModel()
    mutated_model = replace_activations(model)
    # Check that the model now contains a different activation function
    assert 'relu' in str(mutated_model) and 'tanh' in str(mutated_model), "Activation function was not replaced by replace_activations."

def test_uniform_fuzz():
    model = TestModel()
    mutated_model = uniform_fuzz_weight(model)
    # Check that all weights have been perturbed
    all_perturbed = all(torch.any(param.grad != 0).item() for name, param in mutated_model.named_parameters())
    assert all_perturbed, "Not all parameters were perturbed by uniform_fuzz_weight."

# Run all tests
if __name__ == "__main__":
    test_gaussian_fuzzing()
    test_random_shuffle()
    test_remove_activations()
    test_replace_activations()
    test_uniform_fuzz()
    print("All tests passed!")
