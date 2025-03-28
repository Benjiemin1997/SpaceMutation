import torch
import torch.nn as nn
import random
from mut.guassian_fuzz_splayers import gaussian_fuzzing_splayer
from mut.random_shuffle import random_shuffle_weight
from mut.remove_activation import remove_activations
from mut.replace_activation import replace_activations
from mut.uniform_fuzz import uniform_fuzz_weight

def test_model(model, input_data, expected_output=None, device="cuda" if torch.cuda.is_available() else "cpu"):
    model.to(device)
    input_data = input_data.to(device)
    
    # Fuzzing using Gaussian Fuzzing
    gaussian_fuzzing_splayer(model, sigma=0.1)
    
    # Randomly shuffle weights
    random.shuffle_weight(model)
    
    # Remove all activations
    remove_activations(model)
    
    # Replace activations randomly
    replace_activations(model)
    
    # Uniformly fuzz weights
    uniform_fuzz_weight(model, min_val=-0.1, max_val=0.1)
    
    output = model(input_data)
    if expected_output is not None:
        assert torch.allclose(output, expected_output, atol=1e-3, rtol=1e-3), f"Expected {expected_output} but got {output}"
    print("Test passed!")

