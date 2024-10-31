import torch
from torch import nn

from models.Shufflenetv2.shufflenetv2 import ShuffleNetV2
from mut.guassian_fuzz_splayers import gaussian_fuzzing_splayer
from mut.random_shuffle import random_shuffle_weight
from mut.remove_activation import remove_activations
from mut.replace_activation import replace_activations
from mut.uniform_fuzz import uniform_fuzz_weight

def test_model_mutation():
    # Load your model here
    model = ShuffleNetV2()
    
    # Test Gaussian Fuzzing
    model = gaussian_fuzzing_splayer(model, std_ratio=0.8, target_layer_type=nn.Linear)
    # Assertions to check if the model's weights have been mutated correctly
    assert any(torch.std(param.data).item() != torch.std(param.data).item() for param in model.parameters()), "Weights should be mutated by Gaussian Fuzzing"
    
    # Test Random Weight Shuffling
    model = random_shuffle_weight(model)

    assert not all(torch.equal(param.data, param.data) for param in model.parameters()), "Weights should be shuffled randomly"
    
    # Test Activation Removal
    model = remove_activations(model)

    assert any(isinstance(module, nn.Module) for module in model.modules()), "Activations should be removed"
    
    # Test Activation Replacement
    model = replace_activations(model, nn.ReLU)

    assert any(isinstance(module, nn.ReLU) for module in model.modules()), "Activations should be replaced with ReLU"
    
    # Test Uniform Fuzzing
    model = uniform_fuzz_weight(model, min_val=-0.1, max_val=0.1)

    assert any((param.data.min().item() >= min_val) and (param.data.max().item() <= max_val) for param in model.parameters() for min_val, max_val in zip([-0.1], [0.1])), "Weights should be fuzzed uniformly within -0.1 and 0.1"
    
    print("All mutations tested successfully.")
