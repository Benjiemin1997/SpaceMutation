import torch
from torch import nn

from models.VGG16.model_vgg16 import VGG16
from mut.guassian_fuzz_splayers import gaussian_fuzzing_splayer
from mut.random_shuffle import random_shuffle_weight
from mut.remove_activation import remove_activations
from mut.replace_activation import replace_activations
from mut.uniform_fuzz import uniform_fuzz_weight

def test_fgsm_fuzz_weight_model():
    model = VGG16()

    model = uniform_fuzz_weight(model)

    for name, module in model.named_modules():
        if isinstance(module, nn.ReLU) or isinstance(module, nn.PReLU) or isinstance(module, nn.SELU):
            assert not isinstance(module, nn.Identity), f"Activation {name} should be replaced or removed."


    model = gaussian_fuzzing_splayer(model)


    for param in model.parameters():
        assert not torch.allclose(param, torch.zeros_like(param)), "Weights should be different after Gaussian fuzzing."


    model = random_shuffle_weight(model)

    for param in model.parameters():
        assert not torch.allclose(param, torch.roll(param, shifts=1, dims=param.dim())), "Weights should be shuffled after random shuffle."

    model = uniform_fuzz_weight(model)
    for param in model.parameters():
        assert not torch.allclose(param, torch.ones_like(param)), "Weights should be different after uniform fuzzing."

    model = remove_activations(model)

    for name, module in model.named_modules():
        if isinstance(module, nn.ReLU) or isinstance(module, nn.PReLU) or isinstance(module, nn.SELU):
            assert isinstance(module, nn.Identity), f"Activation {name} should be removed."


    model = replace_activations(model)


    for name, module in model.named_modules():
        if isinstance(module, nn.ReLU) or isinstance(module, nn.PReLU) or isinstance(module, nn.SELU):
            assert not isinstance(module, nn.Identity), f"Activation {name} should be replaced."