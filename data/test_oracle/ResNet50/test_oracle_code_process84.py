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

    test_cases = [

        lambda: not torch.equal(mutated_model.fc[0].weight, model.fc[0].weight),

        lambda: sum(torch.sum(mutated_model.conv1.weight != 0) == 0) / mutated_model.conv1.in_channels <= 6,

    ]


    for test_case in test_cases:
        result = test_case()
        assert result, f"Test failed: {test_case.__name__}"

    print("All tests passed successfully.")

if __name__ == "__main__":
    test_neuron_effect_block()