import torch
import torch.nn as nn
import random

from models.Shufflenetv2.shufflenetv2 import ShuffleNetV2
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
    replace_activations(model, activations=[nn.ReLU(), nn.LeakyReLU(), nn.Sigmoid(), nn.Tanh(), nn.ELU(), nn.PReLU(), nn.SELU(), nn.GELU()])
    
    # Uniform fuzzing of weights
    uniform_fuzz_weight(model, min_val=-1, max_val=1)
    
    output = model(input_data)
    
    if expected_output is not None:
        assert torch.allclose(output, expected_output, atol=1e-3, rtol=1e-3), "Output does not match expected output."
    else:
        print(f"Model output: {output}")
        
    print("Test passed successfully!")

# Example usage
input_tensor = torch.randn(1, 3, 224, 224).to("cuda") if torch.cuda.is_available() else torch.randn(1, 3, 224, 224)
model = ShuffleNetV2().to("cuda") if torch.cuda.is_available() else ShuffleNetV2()
test_model(model, input_tensor)
