import torch

from mut.random_shuffle import random_shuffle_weight


# Test Oracle Code

def test_random_shuffle_weight():
    # Create a mock model to test
    class MockModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = torch.nn.Conv2d(1, 6, kernel_size=(5, 5), stride=(1, 1))
            self.maxpool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
            self.linear1 = torch.nn.Linear(256, 120)
            self.linear2 = torch.nn.Linear(120, 84)
            self.linear3 = torch.nn.Linear(84, 10)

        def forward(self, x):
            x = torch.relu(self.conv1(x))
            x = self.maxpool1(x)
            x = torch.flatten(x, start_dim=1)
            x = torch.relu(self.linear1(x))
            x = torch.relu(self.linear2(x))
            x = self.linear3(x)
            return x

    # Initialize the mock model
    model = MockModel()

    # Prepare input data for testing
    input_data = torch.randn(1, 1, 32, 32)  # Assuming input size is 32x32x1

    # Call the function to be tested
    mutated_model = random_shuffle_weight(model)

    # Assertions to verify the output of the function
    assert isinstance(mutated_model, torch.nn.Module), "The returned object is not an instance of torch.nn.Module"
    assert len(mutated_model.children()) == len(model.children()), "The number of children modules has changed after shuffling weights"

    # Verify that the weights have been shuffled by comparing the original and mutated model parameters
    for original_param, mutated_param in zip(model.parameters(), mutated_model.parameters()):
        assert not torch.equal(original_param.data, mutated_param.data), "Weights have not been shuffled"

    print("All tests passed successfully.")
