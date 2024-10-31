import torch
from torch import nn
from mut.guassian_fuzz_splayers import gaussian_fuzzing_splayer
from mut.random_shuffle import random_shuffle_weight
from mut.remove_activation import remove_activations
from mut.replace_activation import replace_activations
from mut.uniform_fuzz import uniform_fuzz_weight

# Define a simple test case for a model
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.conv = nn.Conv2d(3, 6, kernel_size=3, padding=1)

    def forward(self, x):
        return self.conv(x)

# Test case
def test_model_mutations(model, test_input):
    # Gaussian Fuzzing
    model_gaussian = gaussian_fuzzing_splayer(model, std_ratio=0.5)
    output_gaussian = model_gaussian(test_input)
    
    # Random Weight Shuffling
    model_random_shuffle = random_shuffle_weight(model)
    output_random_shuffle = model_random_shuffle(test_input)
    
    # Remove Activations
    model_no_activations = remove_activations(model)
    output_no_activations = model_no_activations(test_input)
    
    # Replace Activations
    model_replaced_activations = replace_activations(model, 'relu', 'elu')
    output_replaced_activations = model_replaced_activations(test_input)
    
    # Uniform Fuzzing
    model_uniform_fuzz = uniform_fuzz_weight(model, low=-0.5, high=0.5)
    output_uniform_fuzz = model_uniform_fuzz(test_input)
    
    # Assertions
    assert not torch.allclose(output_gaussian, output_uniform_fuzz), "Gaussian fuzzing should produce different outputs."
    assert not torch.allclose(output_random_shuffle, output_uniform_fuzz), "Random weight shuffling should produce different outputs."
    assert not torch.allclose(output_no_activations, output_uniform_fuzz), "Removing activations should produce different outputs."
    assert not torch.allclose(output_replaced_activations, output_uniform_fuzz), "Replacing activations should produce different outputs."

# Create a test input tensor
test_input = torch.randn(1, 3, 32, 32)

# Test the model with different mutation techniques
test_model_mutations(SimpleModel(), test_input)