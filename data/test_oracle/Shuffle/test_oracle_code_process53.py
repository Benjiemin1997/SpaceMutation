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

    # Test assertion for inplace flag of ReLU modules
    for name, module in reversed_model.named_children():
        if isinstance(module, nn.ReLU):
            assert not module.inplace, f"Inplace flag should be False after reverse activations, but got {module.inplace}"

    # Test assertion for output of model on input data
    input_data = torch.randn(1, 3, 224, 224).to(device)
    original_output = model(input_data).abs().sum().item()
    reversed_output = reversed_model(input_data).abs().sum().item()

    # Assuming the output should be negative due to the reversed activation
    assert reversed_output < 0, "Output should be negative after reversing activations"

    print("All tests passed successfully.")
