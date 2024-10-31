import torch
from torch import nn

from models.VGG16.model_vgg16 import VGG16
from mut.guassian_fuzz_splayers import gaussian_fuzzing_splayer
from mut.random_shuffle import random_shuffle_weight
from mut.remove_activation import remove_activations
from mut.replace_activation import replace_activations
from mut.uniform_fuzz import uniform_fuzz_weight

def test_gaussian_fuzzing_splayer():
    model = VGG16()


    model_fuzzed = gaussian_fuzzing_splayer(model, std_ratio=0.5, target_layer_type=nn.Linear)


    assert len(list(model_fuzzed.named_modules())) != len(list(model.named_modules())), "Model was not modified"


    for name, layer in model_fuzzed.named_modules():
        if isinstance(layer, nn.Linear):
            for param in layer.parameters():
                assert not torch.allclose(param, model.state_dict()[name][param.name]), f"Layer {name} parameters were not modified"

    print("Gaussian Fuzzing Splayer test passed.")

def test_random_shuffle_weight():

    linear_layer = nn.Linear(10, 10)
    original_weights = linear_layer.weight.clone()


    random_shuffle_weight(linear_layer)


    assert not torch.equal(original_weights, linear_layer.weight), "Weights were not shuffled"

    print("Random Shuffle Weight test passed.")

def test_remove_activations():

    model = VGG16()


    model_no_activations = remove_activations(model)


    assert len(list(model_no_activations.named_modules())) != len(list(model.named_modules())), "Activations were not removed"

    print("Remove Activations test passed.")

def test_replace_activations():

    model = VGG16()

    model_replaced_activations = replace_activations(model)


    assert len(list(model_replaced_activations.named_modules())) != len(list(model.named_modules())), "Activations were not replaced"

    print("Replace Activations test passed.")

def test_uniform_fuzz_weight():

    linear_layer = nn.Linear(10, 10)
    original_weights = linear_layer.weight.clone()


    uniform_fuzz_weight(linear_layer)


    assert not torch.equal(original_weights, linear_layer.weight), "Weights were not fuzzed"

    print("Uniform Fuzz Weight test passed.")
