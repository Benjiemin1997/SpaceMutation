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

    # Prepare input data
    input_data = torch.randn(1, 1, 32, 32)

    # Expected behavior assertion
    original_output = model(input_data)
    
    # Apply the mutation
    mutated_model = random_shuffle_weight(model)

    # Post-mutation behavior assertion
    mutated_output = mutated_model(input_data)
    assert not torch.allclose(original_output, mutated_output), "The model output did not change after applying the mutation."
