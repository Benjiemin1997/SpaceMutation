import torch
import torch.nn as nn
import random

from models.Shufflenetv2.shufflenetv2 import ShuffleNetV2
from mut.guassian_fuzz_splayers import gaussian_fuzzing_splayer
from mut.random_add_activation import add_activation
from mut.random_shuffle import random_shuffle_weight
from mut.remove_activation import remove_activations
from mut.replace_activation import replace_activations
from mut.uniform_fuzz import uniform_fuzz_weight

def test_model(model, input_data, expected_output=None, device='cuda'):
    """
    Test the model with given input data and optionally expected output.
    """
    model.eval() # Set the model to evaluation mode
    with torch.no_grad():
        output = model(input_data)
        if expected_output is not None:
            assert torch.allclose(output, expected_output, atol=1e-3, rtol=1e-3), "Output mismatch"
        else:
            print(f"Model output: {output}")

def test_add_activation(model):
    """
    Test that add_activation modifies the model correctly by adding an activation layer randomly.
    """
    # Apply the mutation
    mutated_model = add_activation(model)

    # Test the model before and after mutation
    original_model = model
    test_model(original_model, torch.randn(1, 3, 32, 32), expected_output=None)
    test_model(mutated_model, torch.randn(1, 3, 32, 32), expected_output=None)

    # Check if any of the layers have been mutated
    mutated_layers = [module for module in mutated_model.modules() 
                      if isinstance(module, nn.Sequential) and len(module) > 2]
    assert mutated_layers, "No layers were mutated by add_activation"

def test_mutation_methods():
    """
    Test the mutation methods (gaussian_fuzzing_splayer, random_shuffle_weight, remove_activations, 
    replace_activations, uniform_fuzz_weight) to ensure they modify the model as intended.
    """
    model = ShuffleNetV2().to('cuda')
    input_data = torch.randn(1, 3, 32, 32).to('cuda')

    # Test gaussian_fuzzing_splayer
    mutated_model = gaussian_fuzzing_splayer(model, input_data)
    test_model(mutated_model, input_data)

    # Test random_shuffle_weight
    mutated_model = random_shuffle_weight(model)
    test_model(mutated_model, input_data)

    # Test remove_activations
    mutated_model = remove_activations(model)
    test_model(mutated_model, input_data)

    # Test replace_activations
    mutated_model = replace_activations(model)
    test_model(mutated_model, input_data)

    # Test uniform_fuzz_weight
    mutated_model = uniform_fuzz_weight(model)
    test_model(mutated_model, input_data)

if __name__ == "__main__":
    test_add_activation(ShuffleNetV2().to('cuda'))
    test_mutation_methods()
