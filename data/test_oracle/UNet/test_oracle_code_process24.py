import torch
from torch import nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def reverse_activations(model):
    for name, module in model.named_children():
        if isinstance(module, nn.ReLU):
            module.inplace = False
            prev_module = list(model.named_children())[list(model.named_children()).index((name, module)) - 1][1]
            prev_module.register_forward_hook(lambda module, input, output: -output)
    return model

# Test case
def test_reverse_activations():
    # Create a simple test model
    class SimpleModel(nn.Module):
        def __init__(self):
            super(SimpleModel, self).__init__()
            self.conv1 = nn.Conv2d(1, 1, 1)
            self.relu1 = nn.ReLU()
            self.conv2 = nn.Conv2d(1, 1, 1)

        def forward(self, x):
            x = self.conv1(x)
            x = self.relu1(x)
            x = self.conv2(x)
            return x

    # Initialize the model
    test_model = SimpleModel()

    # Apply reverse activations
    reversed_model = reverse_activations(test_model)

    # Check that the activation is reversed
    input_data = torch.randn(1, 1, 3, 3)
    original_output = test_model(input_data)
    reversed_output = reversed_model(input_data)

    assert torch.allclose(original_output, -reversed_output), "Activation was not reversed correctly"

    print("Reverse activations test passed!")

# Run the test
test_reverse_activations()
