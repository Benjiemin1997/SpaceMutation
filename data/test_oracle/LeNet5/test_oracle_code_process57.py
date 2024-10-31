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
    class SimpleModel(nn.Module):
        def __init__(self):
            super(SimpleModel, self).__init__()
            self.conv = nn.Conv2d(1, 1, 1)
            self.relu = nn.ReLU()

        def forward(self, x):
            x = self.conv(x)
            x = self.relu(x)
            return x

    model = SimpleModel()
    original_model = model.state_dict().copy()

    # Apply reverse activations
    modified_model = reverse_activations(model)


    assert not any(module.inplace for _, module in modified_model.named_modules()), "Inplace attribute should be False"
    input_data = torch.randn(1, 1, 1, 1)
    output_original = model(input_data)
    output_modified = modified_model(input_data)
    assert torch.allclose(-output_original, output_modified), "Output should be the negative of the original output"

    # Fuzzing test cases
    gaussian_fuzzing_splayer(modified_model, 0.1)
    random_shuffle_weight(modified_model, 0.5)
    remove_activations(modified_model)
    replace_activations(modified_model)
    uniform_fuzz_weight(modified_model, 0.2)

    # Ensure model still works after mutations
    output_fuzzed = modified_model(input_data)
    assert torch.allclose(output_fuzzed, -output_original), "Model should still produce correct output after fuzzing"

    print("All tests passed successfully!")
