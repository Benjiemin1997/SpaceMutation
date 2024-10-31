import torch
from torch import nn
from mut.guassian_fuzz_splayers import gaussian_fuzzing_splayer


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

    # Test if the inplace flag is set to False
    for name, module in modified_model.named_modules():
        if isinstance(module, nn.ReLU):
            assert not module.inplace, f"Inplace flag of ReLU in {name} should be False"

    # Test if the activation has been reversed
    input_data = torch.randn(1, 1, 3, 3)
    original_output = model(input_data)
    modified_output = modified_model(input_data)

    # Assert outputs are negative since ReLU has been reversed
    assert torch.all(modified_output < 0), "Output should be negative after reversing ReLU activation"
    
    # Check that the model structure is modified
    assert original_model != modified_model.state_dict(), "Model structure should be modified"

    # Optional: Apply some mutation techniques before testing
    mutated_model = gaussian_fuzzing_splayer(modified_model)
    assert mutated_model != modified_model, "Mutation should result in a different model"
    

