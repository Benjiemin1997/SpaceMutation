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

    # Test assertion: Check if inplace attribute is set to False
    assert not next(modified_model.children()).inplace, "Inplace attribute should be False after reversing activations"

    # Test assertion: Check if activation function has been reversed
    for name, module in modified_model.named_modules():
        if 'relu' in name:
            assert torch.allclose(module(input=torch.rand(1, 1, 1, 1)), output=torch.zeros(1, 1, 1, 1)), "Activation function should be reversed"

    # Optional: Fuzzing tests
    # Gaussian fuzzing
    gaussian_fuzzing_splayer(modified_model)
    
    # Random shuffle weights
    random_shuffle_weight(modified_model)
    
    # Remove activations
    remove_activations(modified_model)
    
    # Replace activations
    replace_activations(modified_model, nn.LeakyReLU())
    
    # Uniform fuzzing
    uniform_fuzz_weight(modified_model)
    
    # Check that the model has been altered as expected
    assert not torch.equal(original_model, model.state_dict()), "Model should have been altered during fuzzing processes"
