import torch

from torch import nn



def gaussian_fuzzing_splayer(model, std_ratio=0.8, target_layer_type=nn.Linear):
    def layer_wise_perturbation(weights, std_ratio=1):
        std = torch.std(weights) * std_ratio
        noise = torch.normal(0, std, size=weights.size()).to(weights)
        return weights + noise
    with torch.no_grad():
        for name, layer in model.named_modules():
            if isinstance(layer, target_layer_type):
                for param in layer.parameters():
                    if param.requires_grad:
                        param.data = layer_wise_perturbation(param.data, std_ratio)
    return model

