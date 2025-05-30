import torch
from torch import nn
from mut.guassian_fuzz_splayers import gaussian_fuzzing_splayer
from mut.random_shuffle import random_shuffle_weight
from mut.remove_activation import remove_activations
from mut.replace_activation import replace_activations
from mut.uniform_fuzz import uniform_fuzz_weight

def test_oracle(model, test_cases, expected_outputs, test_methods=None):

    # Loop over each test case and apply each mutation method
    if test_methods is None:
        test_methods = [gaussian_fuzzing_splayer, random_shuffle_weight, remove_activations, replace_activations,
                        uniform_fuzz_weight]
    for test_case in test_cases:
        for method in test_methods:
            mutated_model = method(model)


            output = mutated_model(test_case['input'])


            assert torch.allclose(output, torch.tensor(expected_outputs[test_case['name']]), atol=1e-4, rtol=1e-4), f"Test case {test_case['name']} failed after applying {method.__name__} mutation."

# Define test cases and expected outputs
test_cases = [
    {"name": "case1", "input": torch.randn(1, 3, 224, 224)},
    {"name": "case2", "input": torch.randn(2, 3, 224, 224)},
]

expected_outputs = {
    "case1": torch.randn(1, 100),
    "case2": torch.randn(2, 100),
}

