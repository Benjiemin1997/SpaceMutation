import torch

from mut.random_shuffle import random_shuffle_weight


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
    model = random_shuffle_weight(model)

    input_data = torch.randn(1, 1, 32, 32)
    expected_output_shape = (1, 10)

    # Check if the model has been modified
    assert len([p for p in model.parameters() if 'weight' in p.name]) > 0

    # Check if the output shape of the model is as expected after applying the weight shuffle
    output = model(input_data)
    assert output.shape == expected_output_shape, f"Output shape mismatch: Expected {expected_output_shape}, got {output.shape}"

    print("Test Random Shuffle Weight Passed")

if __name__ == "__main__":
    test_random_shuffle_weight()