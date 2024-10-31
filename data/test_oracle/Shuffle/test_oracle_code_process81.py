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

# Test Random Weight Shuffle
random_shuffled_model = random_shuffle_weight(model)
assert not torch.allclose(model.conv.weight, random_shuffled_model.conv.weight), "Weight shuffle did not occur"

# Test Activation Removal
activation_removed_model = remove_activations(model)
assert 'relu' not in str(activation_removed_model), "Activation was not removed"

# Test Activation Replacement
replacement_model = replace_activations(model, {'relu': nn.Tanh()})
assert isinstance(replacement_model.conv1.relu, nn.Tanh), "Activation replacement failed"

# Test Uniform Fuzzing
uniform_fuzzed_model = uniform_fuzz_weight(model, min_val=-0.5, max_val=0.5)
assert (uniform_fuzzed_model.conv.weight < -0.5).sum().item() == 0 and (uniform_fuzzed_model.conv.weight > 0.5).sum().item() == 0, "Uniform fuzzing did not limit weight values within bounds"
