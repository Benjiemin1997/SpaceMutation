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
    model = nn.Sequential(
        nn.Conv2d(1, 6, kernel_size=(5, 5)),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2),
        nn.Flatten(),
        nn.Linear(256, 120),
        nn.ReLU(),
        nn.Linear(120, 84),
        nn.PReLU(),
        nn.Linear(84, 10)
    )

    # Apply reverse activation
    reversed_model = reverse_activations(model)

    # Check if inplace attribute is False after applying reverse activation
    for name, module in reversed_model.named_children():
        if isinstance(module, nn.ReLU):
            assert not module.inplace, f"Inplace attribute of ReLU should be False after reverse activation, but got {module.inplace}"

    # Fuzzing test cases
    fuzzed_models = [
        gaussian_fuzzing_splayer(reversed_model, 0.1),
        random_shuffle_weight(reversed_model),
        remove_activations(reversed_model),
        replace_activations(reversed_model, nn.Tanh()),
        uniform_fuzz_weight(reversed_model, 0.2)
    ]

    # Check all fuzzed models
    for fuzzed_model in fuzzed_models:
        # Ensure model is still callable
        input_data = torch.randn(1, 1, 32, 32).to(fuzzed_model.device)
        output_data = fuzzed_model(input_data)
        assert torch.is_tensor(output_data), "Fuzzed model should return a tensor"

        # Ensure model's output is close to expected values
        expected_output = model(input_data)
        assert torch.allclose(output_data, expected_output, atol=1e-3), "Fuzzed model output does not match original model output"