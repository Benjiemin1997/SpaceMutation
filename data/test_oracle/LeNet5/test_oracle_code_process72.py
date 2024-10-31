import torch

from mut.random_shuffle import random_shuffle_weight


# Test Oracle Code

def test_random_shuffle_weight():
    # Initialize a mock model to test the function on
    mock_model = torch.nn.Sequential(
        torch.nn.Conv2d(1, 6, kernel_size=(5, 5)),
        torch.nn.GELU(),
        torch.nn.MaxPool2d(kernel_size=2),
        torch.nn.Conv2d(6, 16, kernel_size=(5, 5)),
        torch.nn.MaxPool2d(kernel_size=2),
        torch.nn.Flatten(),
        torch.nn.Linear(256, 120),
        torch.nn.Sequential(torch.nn.Linear(120, 84), torch.nn.PReLU()),
        torch.nn.Linear(84, 10)
    )

    # Expected state before shuffling
    expected_state_before = mock_model.state_dict()

    # Apply the function to the mock model
    modified_model = random_shuffle_weight(mock_model)

    # Expected state after shuffling
    expected_state_after = {k: v.clone().detach() for k, v in expected_state_before.items()}
    for param in modified_model.parameters():
        if param.requires_grad:
            expected_state_after[param.name] = param.data.clone().detach()

    # Check that the state of the model has been shuffled
    assert not torch.allclose(expected_state_before['0.0.0.weight'], expected_state_after['0.0.0.weight']), \
        "Weights have not been shuffled properly."
    assert not torch.allclose(expected_state_before['1.0.weight'], expected_state_after['1.0.weight']), \
        "Weights have not been shuffled properly."

    # Check that the model's state dict has been updated
    assert not torch.equal(modified_model.state_dict(), mock_model.state_dict()), \
        "Model state dict has not been updated after shuffling."

    print("Test Random Shuffle Weight Passed.")