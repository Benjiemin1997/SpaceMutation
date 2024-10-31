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

    # Test assertion for forward hook effect on input data
    # Here we would need to have some input data to test the forward hooks properly.
    # This is just a placeholder for demonstration purposes.
    input_data = torch.randn(1, 3, 224, 224)
    output_before = model(input_data)
    output_after = reversed_model(input_data)
    
    # Assuming that reversing activations should result in the negative of the original output,
    # we can use this as a simple test case. This is highly simplified and may not cover all scenarios.
    assert torch.allclose(-output_before, output_after), "Reversing activations did not produce expected results"

    print("All tests passed successfully.")