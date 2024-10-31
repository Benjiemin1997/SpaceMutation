import torch
from torch import nn

from models.VGG16.model_vgg16 import VGG16
from mut.guassian_fuzz_splayers import gaussian_fuzzing_splayer
from mut.random_shuffle import random_shuffle_weight
from mut.remove_activation import remove_activations
from mut.replace_activation import replace_activations
from mut.uniform_fuzz import uniform_fuzz_weight

def test_fgsm_fuzz_weight():
    # Test setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VGG16().to(device)


    mutated_model = random_shuffle_weight(model)


    assert isinstance(mutated_model, nn.Module)
    assert hasattr(mutated_model, 'vgg')
    assert hasattr(mutated_model.vgg.features[0], 'weight')
    assert torch.is_tensor(mutated_model.vgg.features[0].weight.grad)
    

    activations_removed = remove_activations(mutated_model)
    assert isinstance(activations_removed, list) and len(activations_removed) > 0
    

    activations_replaced = replace_activations(mutated_model)
    assert isinstance(activations_replaced, dict) and len(activations_replaced) > 0

    fuzzed_weights = gaussian_fuzzing_splayer(mutated_model)
    assert isinstance(fuzzed_weights, dict) and len(fuzzed_weights) > 0
    

    fuzzed_weights_uniform = uniform_fuzz_weight(mutated_model)
    assert isinstance(fuzzed_weights_uniform, dict) and len(fuzzed_weights_uniform) > 0
    

    shuffled_weights = random_shuffle_weight(mutated_model)
    assert isinstance(shuffled_weights, dict) and len(shuffled_weights) > 0


test_fgsm_fuzz_weight()
