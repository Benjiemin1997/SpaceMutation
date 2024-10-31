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
    class TestModel(nn.Module):
        def __init__(self):
            super(TestModel, self).__init__()
            self.conv = nn.Conv2d(3, 24, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            self.relu = nn.ReLU(inplace=True)
            self.batch_norm = nn.BatchNorm2d(24)

        def forward(self, x):
            x = self.conv(x)
            x = self.relu(x)
            x = self.batch_norm(x)
            return x

    # Create an instance of the model
    model = TestModel()

    # Apply reverse_activations
    reversed_model = reverse_activations(model)

    # Prepare inputs
    input_data = torch.randn(1, 3, 32, 32)

    # Forward pass before and after reversing activations
    original_output = model(input_data)
    reversed_output = reversed_model(input_data)

    # Check if the outputs are opposite in sign
    assert torch.allclose(-original_output, reversed_output), "The reversed activations did not change the output sign."

    print("Test passed successfully!")
