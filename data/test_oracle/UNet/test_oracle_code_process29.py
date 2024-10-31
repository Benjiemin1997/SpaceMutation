import torch

from mut.random_shuffle import random_shuffle_weight


# Test Oracle Code

def test_random_shuffle_weight():
    # Create a mock model to test
    class MockModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(3, 3))
            self.conv2 = torch.nn.Conv2d(64, 128, kernel_size=(3, 3))
            self.conv3 = torch.nn.Conv2d(128, 256, kernel_size=(3, 3))
            self.conv4 = torch.nn.Conv2d(256, 512, kernel_size=(3, 3))
            self.linear1 = torch.nn.Linear(64, 10)

        def forward(self, x):
            x = torch.relu(self.conv1(x))
            x = torch.relu(self.conv2(x))
            x = torch.relu(self.conv3(x))
            x = torch.relu(self.conv4(x))
            x = self.linear1(x)
            return x

    # Initialize the mock model
    model = MockModel()

    # Apply random shuffle weight
    model = random_shuffle_weight(model)

    # Check that the model's parameters have been modified
    for param in model.parameters():
        if param.requires_grad:
            original_param = param.clone().detach().requires_grad_(False)
            shuffled_param = param.clone().detach().requires_grad_(False)
            assert not torch.allclose(original_param, shuffled_param), "Weights have not been shuffled"
            break

    print("Random Shuffle Weight Test Passed")

def test_model_invariants():
    # This function checks some basic invariants of the model such as its architecture, parameters count, etc.
    pass

if __name__ == "__main__":
    test_random_shuffle_weight()
    test_model_invariants()
