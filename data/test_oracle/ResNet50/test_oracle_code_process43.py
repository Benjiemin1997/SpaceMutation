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
            num_neurons = selected_layer.in_features
            neuron_indices = random.sample(range(num_neurons), int(proportion * num_neurons))
            selected_layer.weight[:, neuron_indices] = 0
        elif isinstance(selected_layer, nn.Conv2d):
            num_neurons = selected_layer.out_channels
            neuron_indices = random.sample(range(num_neurons), int(proportion * num_neurons))
            selected_layer.weight[neuron_indices, :, :, :] = 0

    return model

def test_neuron_effect_block():
    from torchvision.models import resnet50
    model = resnet50(pretrained=True)
    mutated_model = neuron_effect_block(model)
    assert isinstance(mutated_model, nn.Module), "The mutated model should be an instance of nn.Module"
    assert len(mutated_model) > 0, "The mutated model should have at least one module"
    for module in mutated_model.modules():
        if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
            assert torch.sum(module.weight.data) != 0, f"Weights of {type(module)} should not be all zeros"
            # Add more checks based on the specifics of your model and the mutation logic

    print("All tests passed successfully.")
