import torch
import torch.nn as nn
import random
from mut.guassian_fuzz_splayers import gaussian_fuzzing_splayer
from mut.random_shuffle import random_shuffle_weight

from mut.uniform_fuzz import uniform_fuzz_weight

def neuron_effect_block(model, proportion=0.1):
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
            num_neurons = selected_layer.in_features
            neuron_indices = random.sample(range(num_neurons), int(proportion * num_neurons))
            selected_layer.weight[:, neuron_indices] = 0
        elif isinstance(selected_layer, nn.Conv2d):
            num_neurons = selected_layer.out_channels
            neuron_indices = random.sample(range(num_neurons), int(proportion * num_neurons))
            selected_layer.weight[neuron_indices, :, :, :] = 0

    return model

def test_neuron_effect_block():
    class SampleModel(nn.Module):
        def __init__(self):
            super(SampleModel, self).__init__()
            self.linear = nn.Linear(10, 10)
            self.conv = nn.Conv2d(3, 3, 3)

        def forward(self, x):
            return self.linear(x) + self.conv(x)

    model = SampleModel()


    mutated_model = neuron_effect_block(model)

    linear_layer = mutated_model.linear
    expected_output = torch.zeros_like(linear_layer.weight)
    gaussian_fuzzing_splayer(linear_layer, expected_output)
    random_shuffle_weight(linear_layer)
    uniform_fuzz_weight(linear_layer)


    conv_layer = mutated_model.conv
    expected_output = torch.zeros_like(conv_layer.weight)
    gaussian_fuzzing_splayer(conv_layer, expected_output)
    random_shuffle_weight(conv_layer)
    uniform_fuzz_weight(conv_layer)

    assert torch.equal(linear_layer.weight, expected_output)
    assert torch.equal(conv_layer.weight, expected_output)