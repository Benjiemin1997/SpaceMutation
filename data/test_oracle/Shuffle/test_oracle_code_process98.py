import random

import torch
import torch.nn as nn

from models.Shufflenetv2.shufflenetv2 import ShuffleNetV2
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

# Test Oracle Code
def test_oracle():
    # Setup
    model = ShuffleNetV2()
    input_data = torch.randn(1, 3, 224, 224)

    # Apply Mutation
    mutated_model = neuron_effect_block(model, proportion=0.5)

    # Fuzzing
    mutated_model.eval()
    mutated_output = mutated_model(input_data)

    # Fuzzing with Gaussian Mutation
    mutated_model = gaussian_fuzzing_splayer(mutated_model, std_dev=0.1)
    mutated_output_gaussian = mutated_model(input_data)

    # Random Shuffling of Weights
    mutated_model = random_shuffle_weight(mutated_model)
    mutated_output_random_shuffled = mutated_model(input_data)

    # Removing Activations
    mutated_model = remove_activations(mutated_model)
    mutated_output_removed_activations = mutated_model(input_data)

    # Replacing Activations
    mutated_model = replace_activations(mutated_model, new_activation=nn.ReLU())
    mutated_output_replaced_activations = mutated_model(input_data)

    # Uniform Fuzzing
    mutated_model = uniform_fuzz_weight(mutated_model, min_val=-0.1, max_val=0.1)
    mutated_output_uniform_fuzzed = mutated_model(input_data)

    # Assertions
    assert not torch.allclose(mutated_output, mutated_output_gaussian), "Fuzzing should change the output."
    assert not torch.allclose(mutated_output, mutated_output_random_shuffled), "Random shuffling should change the output."
    assert not torch.allclose(mutated_output, mutated_output_removed_activations), "Removing activations should change the output."
    assert not torch.allclose(mutated_output, mutated_output_replaced_activations), "Replacing activations should change the output."
    assert not torch.allclose(mutated_output, mutated_output_uniform_fuzzed), "Uniform fuzzing should change the output."

    print("All tests passed successfully!")
