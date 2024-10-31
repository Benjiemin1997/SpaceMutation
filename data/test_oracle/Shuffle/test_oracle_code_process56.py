import torch
import torch.nn as nn

from models.Shufflenetv2.shufflenetv2 import ShuffleNetV2
from mut.guassian_fuzz_splayers import gaussian_fuzzing_splayer
from mut.random_shuffle import random_shuffle_weight
from mut.remove_activation import remove_activations
from mut.replace_activation import replace_activations
from mut.uniform_fuzz import uniform_fuzz_weight

# Define the test oracle
def test_oracle(model, input_data, expected_output, delta=1e-5):
    # Step 1: Run the model on the input data to get the baseline output
    baseline_output = model(input_data)
    
    # Step 2: Apply mutations (fuzzing, shuffling, activation replacement/removal) to the model
    model = gaussian_fuzzing_splayer(model)
    model = random_shuffle_weight(model)
    model = replace_activations(model)
    model = uniform_fuzz_weight(model)
    
    # Step 3: Run the mutated model on the input data and compare the output
    mutated_output = model(input_data)
    
    # Step 4: Assert that the output difference is within the acceptable range
    assert torch.allclose(baseline_output, mutated_output, atol=delta), \
        f"Baseline output {baseline_output} differs significantly from mutated output {mutated_output}"
    
    # Step 5: Optionally, assert other properties of the mutated model, such as its architecture or parameters
    
    # Step 6: Compare the mutated output to the expected output
    assert torch.allclose(mutated_output, expected_output, atol=delta), \
        f"Mutated output {mutated_output} does not match the expected output {expected_output}"

# Example usage
model = ShuffleNetV2()
input_data = torch.randn(1, 3, 224, 224)
expected_output = torch.randn(1, 100)  # Assuming 100 classes for simplicity
test_oracle(model, input_data, expected_output)
