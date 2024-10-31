import torch

from mut.random_shuffle import random_shuffle_weight


# Test Oracle Code
def test_random_shuffle_weight():
    # Create a mock model to test
    class MockModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = torch.nn.Conv2d(3, 24, kernel_size=(3, 3))
            self.bn1 = torch.nn.BatchNorm2d(24)
            self.shuffle_weights = random_shuffle_weight(self)

    # Instantiate the mock model
    mock_model = MockModel()

    # Assert that the model has been correctly initialized
    assert isinstance(mock_model.shuffle_weights, torch.nn.Sequential)

    # Check if the first layer's weights have been shuffled
    assert not torch.allclose(mock_model.conv1.weight, mock_model.shuffle_weights[0].weight)

    # Check if the batch norm layer's weights have been shuffled
    assert not torch.allclose(mock_model.bn1.weight, mock_model.shuffle_weights[1].weight)

    # Check if the shuffled weights are on the correct device
    assert mock_model.shuffle_weights[0].weight.device.type == 'cuda' or mock_model.shuffle_weights[0].weight.device.type == 'cpu'

    print("Random Shuffle Weight Test Passed")
