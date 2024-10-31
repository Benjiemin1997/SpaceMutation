import random

import torch
from torch import nn


def remove_activations(model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    activations = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.ReLU, nn.LeakyReLU, nn.Sigmoid, nn.Tanh)):
            activations.append((name, module))
    if not activations:
        return model
    to_keep = random.choice(activations)

    def get_parent_module(model, child_name):
        names = child_name.split('.')
        parent_name = names[:-1]
        child_name = names[-1]
        parent = model
        for part in parent_name:
            parent = getattr(parent, part)
        return child_name, parent

    for name, module in activations:
        if (name, module) != to_keep:
            parent_name, parent = get_parent_module(model, name)
            setattr(parent, parent_name, nn.Identity())
    return model
