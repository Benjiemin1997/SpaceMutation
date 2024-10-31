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
    mutated_model_gaussian = gaussian_fuzzing_splayer(model, std_ratio=0.8, target_layer_type=nn.Linear)
    assert mutated_model_gaussian is not None, "Gaussian fuzzing should not return None"
    assert any([isinstance(module, nn.Linear) for module in mutated_model_gaussian.modules()]), \
        "The mutated model should contain at least one Linear layer"

    # Test Random Weight Shuffling
    mutated_model_random = random_shuffle_weight(model)
    assert mutated_model_random is not None, "Random weight shuffling should not return None"
    assert any([torch.equal(module.weight, mutated_module.weight) or torch.equal(module.bias, mutated_module.bias) 
                for module, mutated_module in zip(model.modules(), mutated_model_random.modules())]), \
        "At least one weight or bias should be different after shuffling"

    # Test Activation Removal
    mutated_model_no_activations = remove_activations(model)
    assert mutated_model_no_activations is not None, "Activation removal should not return None"
    assert not any([isinstance(module, nn.ReLU) or isinstance(module, nn.PReLU) or isinstance(module, nn.GELU)
                    or isinstance(module, nn.Tanh) for module in mutated_model_no_activations.modules()]), \
        "All activations should be removed from the model"

    # Test Activation Replacement
    mutated_model_replaced_activations = replace_activations(model, nn.ReLU())
    assert mutated_model_replaced_activations is not None, "Activation replacement should not return None"
    assert all([isinstance(module, nn.ReLU) for module in mutated_model_replaced_activations.modules()]), \
        "All activations should be replaced with ReLU"

    # Test Uniform Fuzzing
    mutated_model_uniform = uniform_fuzz_weight(model, scale=0.1)
    assert mutated_model_uniform is not None, "Uniform fuzzing should not return None"
    assert any([torch.any(torch.abs(module.weight - mutated_module.weight) > 0.1) for module, mutated_module in 
                zip(model.modules(), mutated_model_uniform.modules())]), \
        "Weights should have been fuzzed by at least 0.1 magnitude"

if __name__ == "__main__":
    test_model_mutation()
