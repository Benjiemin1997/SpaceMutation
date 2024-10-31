import torch
from torch import nn


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


    assert not next(modified_model.parameters()).inplace, "Inplace attribute should be False after reversing activations"


    for name, module in modified_model.named_modules():
        if isinstance(module, nn.ReLU):
            assert not isinstance(module, nn.ReLU), f"Activation function should not be of type nn.ReLU after reversing activations"


    assert not torch.allclose(original_model, modified_model.state_dict()), "State dict should have changed after reversing activations"


    input_data = torch.randn(1, 1, 3, 3)
    output_data = model(input_data)
    output_reversed = modified_model(input_data)
    assert torch.allclose(output_data * (-1), output_reversed), "Output should be negative when reversed activation is applied"

test_reverse_activations()