import torch
from torch import nn

from models.VGG16.model_vgg16 import VGG16
from mut.guassian_fuzz_splayers import gaussian_fuzzing_splayer
from mut.random_shuffle import random_shuffle_weight



def test_fgsm_fuzz_weight():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VGG16().to(device)

    mutated_model = random_shuffle_weight(model)

    assert isinstance(mutated_model, nn.Module)
    assert hasattr(mutated_model, 'vgg')
    assert hasattr(mutated_model.vgg.features[0], 'weight')
    assert torch.is_tensor(mutated_model.vgg.features[0].weight.grad)

    mutated_model_gaussian = gaussian_fuzzing_splayer(mutated_model)

    assert isinstance(mutated_model_gaussian, nn.Module)
    assert hasattr(mutated_model_gaussian, 'vgg')
    assert hasattr(mutated_model_gaussian.vgg.features[0], 'weight')
    assert torch.is_tensor(mutated_model_gaussian.vgg.features[0].weight.grad)

