import torch
from torch import nn
from mut.guassian_fuzz_splayers import gaussian_fuzzing_splayer
from mut.random_shuffle import random_shuffle_weight
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
def test_model(model, test_input):
    # Test the original model
    output_before = model(test_input)
    print("Original Model Output:", output_before)

    # Apply Gaussian Fuzzing
    model = gaussian_fuzzing_splayer(model, std_ratio=0.5)
    output_after_gaussian = model(test_input)
    print("Gaussian Fuzzed Model Output:", output_after_gaussian)

    # Apply Random Weight Shuffling
    model = random_shuffle_weight(model)
    output_after_random_shuffling = model(test_input)
    print("Randomly Shuffled Model Output:", output_after_random_shuffling)

    # Remove Activations
    model = remove_activations(model)
    output_after_activation_removal = model(test_input)
    print("Model with Activations Removed Output:", output_after_activation_removal)

    # Replace Activations
    model = replace_activations(model, {'ReLU': nn.SELU()})
    output_after_activation_replacement = model(test_input)
    print("Model with Activations Replaced Output:", output_after_activation_replacement)

    # Uniformly Fuzz Weights
    model = uniform_fuzz_weight(model)
    output_after_uniform_fuzzing = model(test_input)
    print("Uniformly Fuzzed Model Output:", output_after_uniform_fuzzing)

    # Assertions to verify that the outputs are not exactly the same after modifications
    assert not torch.allclose(output_before, output_after_gaussian), "Gaussian Fuzzing did not change the output"
    assert not torch.allclose(output_before, output_after_random_shuffling), "Random Shuffling did not change the output"
    assert not torch.allclose(output_before, output_after_activation_removal), "Activations Removal did not change the output"
    assert not torch.allclose(output_before, output_after_activation_replacement), "Activation Replacement did not change the output"
    assert not torch.allclose(output_before, output_after_uniform_fuzzing), "Uniform Fuzzing did not change the output"

# Create a simple model instance
model = SimpleModel()

# Test input
test_input = torch.randn(1, 3, 32, 32)

# Run the test case
test_model(model, test_input)
