import torch

from mut.random_shuffle import random_shuffle_weight


# Test Oracle Code

def test_random_shuffle_weight():
    # Create a mock model to test
    class MockModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = torch.nn.Conv2d(1, 6, kernel_size=(5, 5), stride=(1, 1))
            self.conv2 = torch.nn.Conv2d(6, 16, kernel_size=(5, 5), stride=(1, 1))
            self.linear1 = torch.nn.Linear(256, 120)
            self.linear2 = torch.nn.Linear(120, 84)
            self.linear3 = torch.nn.Linear(84, 10)

        def forward(self, x):
            x = torch.relu(self.conv1(x))
            x = torch.relu(self.conv2(x))
            x = torch.max_pool2d(x, 2, 2)
            x = torch.flatten(x)
            x = torch.relu(self.linear1(x))
            x = torch.relu(self.linear2(x))
            x = torch.relu(self.linear3(x))
            return x

    # Initialize the mock model
    model = MockModel()

    # Apply random shuffle weight
    model = random_shuffle_weight(model)

    # Define expected output shape
    input_data = torch.randn(1, 1, 32, 32)
    expected_output_shape = (1, 10)

    # Check if the model has been modified
    assert len([name for name, _ in model.named_children()]) == 13
    assert all(hasattr(model, attr) for attr in ['conv1', 'conv2', 'linear1', 'linear2', 'linear3'])

    # Check if the output shape matches after applying random shuffle weight
    output = model(input_data)
    assert output.shape == expected_output_shape

    print("Random Shuffle Weight Test Passed.")
