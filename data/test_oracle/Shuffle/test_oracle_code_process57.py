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
def test_model_mutations(model, mutation_functions):
    # Apply each mutation to the model and test it
    for func in mutation_functions:
        mutated_model = func(model)
        # Check that the mutated model is different from the original one
        assert not torch.allclose(model.state_dict(), mutated_model.state_dict())
        # Check that the mutated model works as expected (this part depends on your specific use case)
        input_data = torch.randn(1, 3, 32, 32)
        output_original = model(input_data)
        output_mutated = mutated_model(input_data)
        assert torch.allclose(output_original, output_mutated, atol=1e-3), "Output of mutated model differs from original"

# List of mutation functions to apply to the model
mutation_functions = [
    gaussian_fuzzing_splayer,
    random_shuffle_weight,
    remove_activations,
    replace_activations,
    uniform_fuzz_weight
]

# Create a simple model instance
model = SimpleModel()

# Run the test case
test_model_mutations(model, mutation_functions)
