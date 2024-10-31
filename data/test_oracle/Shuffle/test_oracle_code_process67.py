import torch
import torch.nn as nn
import random
from mut.guassian_fuzz_splayers import gaussian_fuzzing_splayer
from mut.random_shuffle import random_shuffle_weight
from mut.remove_activation import remove_activations
from mut.replace_activation import replace_activations
from mut.uniform_fuzz import uniform_fuzz_weight

def neuron_effect_block(model, proportion=0.5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    modules = list(model.modules())
    linear_layers = [m for m in modules if isinstance(m, nn.Linear)]
    conv_layers = [m for m in modules if isinstance(m, nn.Conv2d)]
    valid_layers = linear_layers + conv_layers

    if not valid_layers:
        raise ValueError("No valid layers found. Ensure the model contains at least one Linear or Conv2d layer.")

    selected_layer = random.choice(valid_layers)

    with torch.no_grad():
        if isinstance(selected_layer, nn.Linear):
            # For Linear layers, we want to affect the input neurons.
            num_neurons = selected_layer.in_features
            neuron_indices = random.sample(range(num_neurons), int(proportion * num_neurons))
            selected_layer.weight[:, neuron_indices] = 0
        elif isinstance(selected_layer, nn.Conv2d):
            # For Conv2d layers, we want to affect the output channels.
            num_neurons = selected_layer.out_channels
            neuron_indices = random.sample(range(num_neurons), int(proportion * num_neurons))
            selected_layer.weight[neuron_indices, :, :, :] = 0

    return model

def test_neuron_effect_block():
    # Create a sample model for testing
    class SampleModel(nn.Module):
        def __init__(self):
            super(SampleModel, self).__init__()
            self.linear = nn.Linear(10, 10)
            self.conv = nn.Conv2d(3, 16, 3, padding=1)
            self.relu = nn.ReLU()

        def forward(self, x):
            x = self.linear(x)
            x = self.conv(x)
            x = self.relu(x)
            return x

    model = SampleModel().eval()

    # Test neuron_effect_block function
    mutated_model = neuron_effect_block(model, proportion=0.3)

    # Check if the model's parameters have been modified correctly
    for name, param in mutated_model.named_parameters():
        if 'weight' in name:
            if 'linear.weight' in name:
                assert torch.sum(param) == 0, f"Linear weights should be all zero after mutation."
            elif 'conv.weight' in name:
                assert torch.sum(param) == 0, f"Conv weights should be all zero after mutation."

    print("Neuron effect block test passed.")