import torch

from mut.random_shuffle import random_shuffle_weight


def test_random_shuffle_weight():
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

    # Expected output before shuffling
    expected_output_before = model(input_data)

    shuffled_model = random_shuffle_weight(model)
    output_after_shuffling = shuffled_model(input_data)

    assert not torch.allclose(expected_output_before, output_after_shuffling), "Output should be different after shuffling weights"

    # Check if the model's parameters have been altered
    for param in shuffled_model.parameters():
        assert not torch.equal(param, model.state_dict()[param.name]), "Parameters should be different after shuffling"