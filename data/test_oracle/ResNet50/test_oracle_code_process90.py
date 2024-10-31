from torch import nn

from models.ResNet50.model_resnet50 import ResNet50
from mut.guassian_fuzz_splayers import gaussian_fuzzing_splayer
from mut.random_shuffle import random_shuffle_weight
from mut.remove_activation import remove_activations
from mut.replace_activation import replace_activations
from mut.uniform_fuzz import uniform_fuzz_weight

def test_model_mutation():
    model = ResNet50()
    mutated_model_gaussian = gaussian_fuzzing_splayer(model, std_ratio=0.5, target_layer_type=nn.Linear)
    assert isinstance(mutated_model_gaussian, nn.Module)
    mutated_model_random_shuffle = random_shuffle_weight(model)
    assert isinstance(mutated_model_random_shuffle, nn.Module)
    mutated_model_remove_activations = remove_activations(model)
    assert isinstance(mutated_model_remove_activations, nn.Module)


    mutated_model_replace_activations = replace_activations(model)
    assert isinstance(mutated_model_replace_activations, nn.Module)


    mutated_model_uniform_fuzz = uniform_fuzz_weight(model)
    assert isinstance(mutated_model_uniform_fuzz, nn.Module)

    print("All mutation tests passed successfully!")
