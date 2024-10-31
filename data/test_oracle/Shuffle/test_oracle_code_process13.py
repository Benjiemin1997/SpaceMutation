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

def test():
    # Test case to ensure the reverse_activations function works as expected.
    # We'll create a simple model with ReLU activations, apply reverse_activations, and verify the outputs.

    class SimpleModel(nn.Module):
        def __init__(self):
            super(SimpleModel, self).__init__()
            self.conv = nn.Conv2d(1, 1, 1)

        def forward(self, x):
            return nn.functional.relu(self.conv(x))

    model = SimpleModel()
    input_data = torch.randn(1, 1, 10, 10)

    original_output = model(input_data)
    reversed_model = reverse_activations(model)
    reversed_output = reversed_model(input_data)


    assert not torch.allclose(original_output, reversed_output), "The activation reversal did not work as expected."
    assert torch.all(torch.ge(reversed_output, 0)), "Some negative values were present after reversing activations."

    print("All tests passed successfully!")
