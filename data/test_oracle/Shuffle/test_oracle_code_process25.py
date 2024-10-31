import torch
import torch.nn as nn
import random

from models.Shufflenetv2.shufflenetv2 import ShuffleNetV2
from mut.guassian_fuzz_splayers import gaussian_fuzzing_splayer
from mut.random_shuffle import random_shuffle_weight
from mut.remove_activation import remove_activations
from mut.replace_activation import replace_activations
from mut.uniform_fuzz import uniform_fuzz_weight

def test_model(model, input_data, expected_output=None, device='cuda'):
    model.to(device)
    input_data = input_data.to(device)
    output = model(input_data)
    
    if expected_output is not None:
        assert torch.allclose(output, expected_output, atol=1e-2), "Model output does not match expected output."
        
    # Testing various mutation methods
    for mutation_method in [gaussian_fuzzing_splayer, random_shuffle_weight, remove_activations, replace_activations, uniform_fuzz_weight]:
        mutated_model = mutation_method(model)
        mutated_output = mutated_model(input_data)
        assert mutated_output.shape == output.shape, "Mutation changed output shape."

def main():
    # Load your model here
    model = ShuffleNetV2()
    
    # Load your test input data here
    input_data = torch.randn(1, 3, 224, 224).to('cuda')
    
    # Call the test function
    test_model(model, input_data)

if __name__ == "__main__":
    main()
