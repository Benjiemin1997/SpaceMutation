import torch
from torch import nn

from models.VGG16.model_vgg16 import VGG16
from mut.guassian_fuzz_splayers import gaussian_fuzzing_splayer
from mut.random_shuffle import random_shuffle_weight
from mut.remove_activation import remove_activations
from mut.replace_activation import replace_activations
from mut.uniform_fuzz import uniform_fuzz_weight

def test_fgsm_fuzz_weight():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VGG16().to(device)
    mutated_model = uniform_fuzz_weight(model)
    assert isinstance(mutated_model, nn.Module)
    assert hasattr(mutated_model, 'vgg')
    assert hasattr(mutated_model.vgg.features[0], 'weight')
    assert torch.is_tensor(mutated_model.vgg.features[0].weight.grad)
    

    mutated_model_gaussian = gaussian_fuzzing_splayer(mutated_model)
    mutated_model_random_shuffle = random_shuffle_weight(mutated_model)
    mutated_model_remove_activations = remove_activations(mutated_model)
    mutated_model_replace_activations = replace_activations(mutated_model)
    mutated_model_uniform_fuzz = uniform_fuzz_weight(mutated_model)


    assert isinstance(mutated_model_gaussian, nn.Module)
    assert isinstance(mutated_model_random_shuffle, nn.Module)
    assert isinstance(mutated_model_remove_activations, nn.Module)
    assert isinstance(mutated_model_replace_activations, nn.Module)
    assert isinstance(mutated_model_uniform_fuzz, nn.Module)
    
