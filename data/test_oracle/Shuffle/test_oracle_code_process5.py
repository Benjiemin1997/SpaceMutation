import torch
from torch import nn

from models.Shufflenetv2.shufflenetv2 import ShuffleNetV2
from mut.guassian_fuzz_splayers import gaussian_fuzzing_splayer
from mut.random_shuffle import random_shuffle_weight
from mut.remove_activation import remove_activations
from mut.replace_activation import replace_activations
from mut.uniform_fuzz import uniform_fuzz_weight

def reverse_activations(model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for name, module in model.named_children():
        if isinstance(module, nn.ReLU):
            module.inplace = False
            prev_module = list(model.named_children())[list(model.named_children()).index((name, module)) - 1][1]
            prev_module.register_forward_hook(lambda module, input, output: -output)
    return model

def test_oracle(model):
    # Test Gaussian Fuzzing Splayer
    gaussian_fuzzing_splayer(model)

    # Test Random Weight Shuffling
    random_shuffle_weight(model)

    # Test Removing Activations
    remove_activations(model)

    # Test Replacing Activations
    replace_activations(model)

    # Test Uniform Weight Fuzzing
    uniform_fuzz_weight(model)

    # Assert model functionality after each mutation
    assert model(torch.randn(1, 3, 32, 32)).shape == torch.Size([1, 100]), "Model output shape is incorrect after mutation."

if __name__ == "__main__":
    # Load your model here
    loaded_model = ShuffleNetV2()
    
    # Run the test oracle on the model
    test_oracle(loaded_model)
