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
    # Create a simple test model
    class TestModel(nn.Module):
        def __init__(self):
            super(TestModel, self).__init__()
            self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
            self.relu1 = nn.ReLU()
            self.pool1 = nn.MaxPool2d(2)
            self.fc1 = nn.Linear(256, 120)
            self.fc2 = nn.Linear(120, 10)

    # Initialize the model
    test_model = TestModel()

    # Apply reverse activation
    reversed_model = reverse_activations(test_model)

    # Check if the activation is reversed
    for name, module in reversed_model.named_children():
        if isinstance(module, nn.ReLU):
            assert not module.inplace, f"Inplace operation still enabled after reversing activations for layer {name}"
            break

    # Apply some mutation operations to the model
    gaussian_fuzzing_splayer(reversed_model)
    random_shuffle_weight(reversed_model)
    remove_activations(reversed_model)
    replace_activations(reversed_model, nn.ReLU, nn.GELU)
    uniform_fuzz_weight(reversed_model)

    # Check that the model has been mutated correctly
    assert not all(isinstance(module, nn.ReLU) for _, module in reversed_model.named_children()), "All ReLU layers have not been mutated"
    assert any(isinstance(module, nn.GELU) for _, module in reversed_model.named_children()), "No GELU layer was introduced after replacing activations"

    print("All tests passed successfully!")
