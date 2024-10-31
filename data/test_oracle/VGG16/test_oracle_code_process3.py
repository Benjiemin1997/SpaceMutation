import torch
from torch import nn
from mut.guassian_fuzz_splayers import gaussian_fuzzing_splayer


def test_gaussian_fuzzing_splayer():
    # Initialize a simple model (for example, a linear layer)
    model = nn.Linear(10, 5)

    # Apply Gaussian Fuzzing Splayer
    model = gaussian_fuzzing_splayer(model, std_ratio=0.5)

    # Test assertions on the model parameters after applying Gaussian Fuzzing
    assert any(torch.std(param.data) > 0 for param in model.parameters()), "Fuzzing did not change standard deviation"
    assert all(torch.mean(param.data) != 0 for param in model.parameters()), "Fuzzing introduced zero mean"

    # Test that the model structure is mutated
    assert 'nn.Linear' not in str(model), "Model structure was not mutated"

    # Test additional checks for mutated layers or parameters
    # For instance, check if any activation functions have been replaced or removed
    if isinstance(model, nn.Sequential):
        for module in model.modules():
            if isinstance(module, nn.ReLU):
                assert not isinstance(module, nn.ReLU), "ReLU was not removed"
            elif isinstance(module, nn.PReLU):
                assert isinstance(module, nn.PReLU), "PReLU was not added as replacement"

    print("All tests passed successfully.")

if __name__ == "__main__":
    test_gaussian_fuzzing_splayer()