import torch
import torch.nn as nn
import random

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
            self.conv = nn.Conv2d(3, 10, 3)
    
    sample_model = SampleModel()

    # Test neuron_effect_block on a linear layer
    neuron_effect_block(sample_model.linear)
    assert not torch.allclose(sample_model.linear.weight, torch.zeros_like(sample_model.linear.weight)), \
        "Linear layer weights should be modified"

    # Test neuron_effect_block on a convolutional layer
    neuron_effect_block(sample_model.conv)
    assert not torch.allclose(sample_model.conv.weight, torch.zeros_like(sample_model.conv.weight)), \
        "Convolutional layer weights should be modified"

    print("All tests passed successfully.")
