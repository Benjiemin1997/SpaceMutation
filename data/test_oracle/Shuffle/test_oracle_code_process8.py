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
    # Create a simple model for testing purposes
    class SimpleModel(nn.Module):
        def __init__(self):
            super(SimpleModel, self).__init__()
            self.linear = nn.Linear(10, 10)
            self.conv = nn.Conv2d(1, 1, 3)
        
        def forward(self, x):
            x = self.linear(x)
            x = self.conv(x)
            return x

    model = SimpleModel()
    
    # Test case 1: Linear layer
    model_copy = neuron_effect_block(model, proportion=0.5)
    assert torch.equal(model.linear.weight, torch.tensor([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                                          [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                                          [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                                          [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                                          [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                                          [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                                          [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                                          [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                                          [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                                          [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]])), "Linear layer weights are not correctly set to zero."

    # Test case 2: Conv2d layer
    model_copy = neuron_effect_block(model, proportion=0.5)
    assert torch.equal(model.conv.weight, torch.tensor([[[[0., 0., 0.],
                                                         [0., 0., 0.],
                                                         [0., 0., 0.]],
                                                        [[0., 0., 0.],
                                                         [0., 0., 0.],
                                                         [0., 0., 0.]],
                                                        [[0., 0., 0.],
                                                         [0., 0., 0.],
                                                         [0., 0., 0.]],
                                                        [[0., 0., 0.],
                                                         [0., 0., 0.],
                                                         [0., 0., 0.]],
                                                        [[0., 0., 0.],
                                                         [0., 0., 0.],
                                                         [0., 0., 0.]],
                                                        [[0., 0., 0.],
                                                         [0., 0., 0.],
                                                         [0., 0., 0.]],
                                                        [[0., 0., 0.],
                                                         [0., 0., 0.],
                                                         [0., 0., 0.]],
                                                        [[0., 0., 0.],
                                                         [0., 0., 0.],
                                                         [0., 0., 0.]],
                                                        [[0., 0., 0.],
                                                         [0., 0., 0.],
                                                         [0., 0., 0.]],
                                                        [[0., 0., 0.],
                                                         [0., 0., 0.],
                                                         [0., 0., 0.]]]])), "Conv2d layer weights are not correctly set to zero."

    print("All tests passed successfully.")