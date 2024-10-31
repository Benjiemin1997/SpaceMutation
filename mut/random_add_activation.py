import torch
import torch.nn as nn
import random


def add_activation(model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    def pick_activation_randomly():
        activations = [
            nn.ReLU(),
            nn.LeakyReLU(),
            nn.Sigmoid(),
            nn.Tanh(),
            nn.ELU(),
            nn.PReLU(),
            nn.SELU(),
            nn.GELU()
        ]
        return random.choice(activations)

    modules = list(model.modules())
    layers = [module for module in modules if isinstance(module, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d))]
    if not layers:
        raise ValueError("The model does not contain any linear or convolutional layers.")
    layer_to_mutate = random.choice(layers)
    activation = pick_activation_randomly()
    new_layer = nn.Sequential(layer_to_mutate, activation)
    parent_module, parent_module_name, layer_name = None, None, None
    for name, module in model.named_modules():
        for child_name, child in module.named_children():
            if child is layer_to_mutate:
                parent_module = module
                parent_module_name = name
                layer_name = child_name
                break
    if parent_module is None:
        raise ValueError("Could not find the parent module of the selected layer.")
    setattr(parent_module, layer_name, new_layer)
    return model
