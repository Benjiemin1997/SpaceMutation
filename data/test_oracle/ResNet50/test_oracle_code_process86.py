import torch
import torch.nn as nn
import random

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
            # For Linear layers, we want to affect the input neurons.
            num_neurons = selected_layer.in_features
            neuron_indices = random.sample(range(num_neurons), int(proportion * num_neurons))
            selected_layer.weight[:, neuron_indices] = 0
        elif isinstance(selected_layer, nn.Conv2d):

            num_neurons = selected_layer.out_channels
            neuron_indices = random.sample(range(num_neurons), int(proportion * num_neurons))
            selected_layer.weight[neuron_indices, :, :, :] = 0

    return model

def test_neuron_effect_block():
    # Create a simple model for testing purposes
    class SimpleModel(nn.Module):
        def __init__(self):
            super(SimpleModel, self).__init__()
            self.linear = nn.Linear(10, 10)
    
        def forward(self, x):
            return self.linear(x)
    
    # Initialize the model
    model = SimpleModel().to(torch.device("cpu"))
    
    # Apply the neuron effect block mutation
    mutated_model = neuron_effect_block(model, proportion=0.5)
    
    # Test cases
    # Case 1: Check if the model's weights have been altered
    assert not torch.allclose(mutated_model.linear.weight, model.linear.weight)
    
    # Case 2: Check if the proportion of affected neurons is approximately correct
    num_original_weights = model.linear.weight.numel()
    num_mutated_weights = mutated_model.linear.weight.numel() - torch.count_nonzero(mutated_model.linear.weight)
    expected_num_mutated_weights = int(num_original_weights * 0.5)
    assert abs(num_mutated_weights - expected_num_mutated_weights) <= 1
    
    print("All tests passed successfully.")

if __name__ == "__main__":
    test_neuron_effect_block()
