import random

import torch
from torch import nn


def replace_activations(model):
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

    def get_parent_module(model, child_name):
        names = child_name.split('.')
        parent_name = names[:-1]
        child_name = names[-1]
        parent = model
        for part in parent_name:
            parent = getattr(parent, part)
        return child_name, parent

    activation = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.ReLU, nn.LeakyReLU, nn.Sigmoid, nn.Tanh, nn.ELU, nn.PReLU, nn.SELU, nn.GELU)):
            activation.append((name, module))
    if not activation:
        return model
    for name, module in activation:
        parent_name, parent = get_parent_module(model, name)
        new_activation = pick_activation_randomly()
        setattr(parent, parent_name, new_activation)
    return model
