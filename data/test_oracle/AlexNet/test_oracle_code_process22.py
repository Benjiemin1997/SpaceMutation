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

    # Test if the gradient is correctly computed after applying reverse activation
    loss = torch.mean(original_output**2)
    loss.backward()
    reversed_loss = torch.mean(reversed_output**2)
    reversed_loss.backward()

    # Check if gradients are opposite in sign
    for param, reversed_param in zip(model.parameters(), reversed_model.parameters()):
        assert torch.allclose(param.grad * (-1), reversed_param.grad)

    print("All tests passed!")

if __name__ == "__main__":
    test_reverse_activations()
