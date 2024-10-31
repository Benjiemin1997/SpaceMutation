import torch
from torch import nn
from mut.guassian_fuzz_splayers import gaussian_fuzzing_splayer
from mut.random_shuffle import random_shuffle_weight
from mut.remove_activation import remove_activations
from mut.replace_activation import replace_activations
from mut.uniform_fuzz import uniform_fuzz_weight

# Define a simple test model for demonstration purposes
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.fc1 = nn.Linear(10*12*12, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = x.view(-1, 10*12*12)
        x = self.fc1(x)
        return x

# Create an instance of the model
model = SimpleModel()

# Test cases

# Test Gaussian Fuzzing
model_gaussian_fuzzed = gaussian_fuzzing_splayer(model, std_ratio=0.5)
assert model_gaussian_fuzzed.conv1.weight.std().item() > 0, "Fuzzing did not alter the weight standard deviation."

# Test Random Weight Shuffling
model_random_shuffled = random_shuffle_weight(model)
assert torch.all(model_random_shuffled.conv1.weight != model.conv1.weight), "Weights were not shuffled."

# Test Activation Removal
model_no_activations = remove_activations(model)
assert not any(isinstance(module, nn.ReLU) or isinstance(module, nn.LeakyReLU) for module in model_no_activations.modules()), "Activations were not removed."

# Test Activation Replacement
model_replaced_activations = replace_activations(model, nn.ReLU())
assert all(isinstance(module, nn.ReLU) for module in model_replaced_activations.modules()), "Activations were not replaced."

# Test Uniform Weight Fuzzing
model_uniform_fuzzed = uniform_fuzz_weight(model, min_val=-0.1, max_val=0.1)
assert torch.all(model_uniform_fuzzed.conv1.weight >= -0.1) and torch.all(model_uniform_fuzzed.conv1.weight <= 0.1), "Weights were not fuzzed within the specified range."
