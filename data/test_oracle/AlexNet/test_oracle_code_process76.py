import torch
from torch import nn
from mut.guassian_fuzz_splayers import gaussian_fuzzing_splayer
from mut.random_shuffle import random_shuffle_weight
from mut.remove_activation import remove_activations
from mut.replace_activation import replace_activations
from mut.uniform_fuzz import uniform_fuzz_weight

def reverse_activations(model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for name, module in model.named_children():
        if isinstance(module, nn.ReLU):
            module.inplace = False
            prev_module = list(model.named_children())[list(model.named_children()).index((name, module)) - 1][1]
            prev_module.register_forward_hook(lambda module, input, output: -output)
    return model

def test_reverse_activations():
    # Create a simple model for testing
    model = nn.Sequential(nn.Linear(10, 10), nn.ReLU(), nn.Linear(10, 1))

    # Test the reverse_activations function
    reversed_model = reverse_activations(model)

    # Apply Gaussian fuzzing to the model's weights
    gaussian_fuzzing_splayer(reversed_model, std_dev=0.1)

    # Apply random shuffle to the model's weights
    random_shuffle_weight(reversed_model)

    # Apply weight removal to the model's weights
    remove_activations(reversed_model)

    # Apply activation replacement to the model's weights
    replace_activations(reversed_model, nn.Tanh())

    # Apply uniform fuzzing to the model's weights
    uniform_fuzz_weight(reversed_model, min_val=-0.1, max_val=0.1)

    # Check if the model's output changes after applying the mutations
    input_data = torch.randn(1, 10)
    original_output = model(input_data).item()
    mutated_output = reversed_model(input_data).item()

    assert original_output != mutated_output, "The model output did not change after applying mutations."
    print("Test passed: Model output changed after applying mutations.")

test_reverse_activations()