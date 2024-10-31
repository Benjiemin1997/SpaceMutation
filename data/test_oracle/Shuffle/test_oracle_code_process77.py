import torch
from torch import nn

from models.Shufflenetv2.shufflenetv2 import ShuffleNetV2


def reverse_activations(model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for name, module in model.named_children():
        if isinstance(module, nn.ReLU):
            module.inplace = False
            prev_module = list(model.named_children())[list(model.named_children()).index((name, module)) - 1][1]
            prev_module.register_forward_hook(lambda module, input, output: -output)
    return model

def test():
    # Test case for reverse_activations function
    model = ShuffleNetV2()

    # Reverse activations
    reversed_model = reverse_activations(model)

    # Test that the output is negative of the original model's output
    input_data = torch.randn(1, 3, 224, 224)
    original_output = model(input_data)
    reversed_output = reversed_model(input_data)

    assert torch.allclose(-original_output, reversed_output), "The reversed model does not reverse the activation correctly."

    print("Test passed successfully.")