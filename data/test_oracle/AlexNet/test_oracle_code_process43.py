import torch
from torch import nn

from mut.reverse_activation import reverse_activations

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def test_reverse_activations():
    # Create a simple model for testing
    model = torch.nn.Sequential(
        torch.nn.Conv2d(3, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2)),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(kernel_size=3, stride=2),
        torch.nn.Conv2d(64, 192, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)),
        torch.nn.Tanh(),
        torch.nn.MaxPool2d(kernel_size=3, stride=2),
        torch.nn.Identity()
    ).to(device)

    # Apply reverse activation
    reversed_model = reverse_activations(model)

    # Test if the forward pass produces the expected result when passing a tensor through it
    input_tensor = torch.randn(1, 3, 224, 224).to(device)
    original_output = model(input_tensor)
    reversed_output = reversed_model(input_tensor)

    # Check if the outputs are equal but with opposite sign
    assert torch.allclose(original_output, -reversed_output)

    # Test if the activation is reversed by trying to apply the original model on the output of the reversed model
    original_output_again = model(reversed_output)
    assert torch.allclose(original_output, -original_output_again)

# Run the test
test_reverse_activations()
