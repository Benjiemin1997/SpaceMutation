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
        self.conv = nn.Conv2d(1, 10, kernel_size=5)
        self.fc = nn.Linear(10 * 12 * 12, 10)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, 10 * 12 * 12)
        x = self.fc(x)
        return x

# Create an instance of the model
model = SimpleModel()

# Test cases

# Test Gaussian Fuzzing
model_gaussian_fuzzed = gaussian_fuzzing_splayer(model, std_ratio=0.5)
assert model_gaussian_fuzzed.conv.weight.std().item() > 0.5, "Fuzzing failed to change weight standard deviation"

# Test Random Weight Shuffling
model_random_shuffled = random_shuffle_weight(model)
for param1, param2 in zip(model.conv.weight, model_random_shuffled.conv.weight):
    assert not torch.allclose(param1, param2), "Random shuffling failed to shuffle weights"

# Test Activation Removal
model_no_activations = remove_activations(model)
assert not hasattr(model_no_activations.fc, 'forward'), "Activation removal did not remove activations"

# Test Activation Replacement
model_replaced_activations = replace_activations(model, nn.ReLU())
assert isinstance(model_replaced_activations.fc.forward.__func__, nn.ReLU), "Activation replacement failed"

# Test Uniform Fuzzing
model_uniform_fuzzed = uniform_fuzz_weight(model, lower_bound=-0.1, upper_bound=0.1)
assert (model_uniform_fuzzed.conv.weight < 0.1).sum().item() > 0 and (model_uniform_fuzzed.conv.weight > -0.1).sum().item() > 0, "Uniform fuzzing failed to apply bounds"