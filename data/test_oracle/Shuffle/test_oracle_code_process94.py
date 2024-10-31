import torch

from mut.random_shuffle import random_shuffle_weight


# Function to test the random_shuffle_weight method
def test_random_shuffle_weight():
    # Create a simple model (for demonstration purposes, this is not a real ShuffleNetV2 model)
    class SimpleModel(torch.nn.Module):
        def __init__(self):
            super(SimpleModel, self).__init__()
            self.conv = torch.nn.Conv2d(3, 1, kernel_size=3)

        def forward(self, x):
            return self.conv(x)

    # Initialize the model
    model = SimpleModel()

    # Define expected behavior or expected output before shuffling weights
    original_output = model(torch.randn(1, 3, 32, 32))

    # Apply the random_shuffle_weight method
    model = random_shuffle_weight(model)

    # Check that the model's parameters have been shuffled
    for param in model.parameters():
        assert not torch.allclose(param, original_output.conv.weight), "Weights were not shuffled"

    # Check that the model still produces an output after shuffling weights
    shuffled_output = model(torch.randn(1, 3, 32, 32))
    assert shuffled_output.shape == original_output.shape, "Output shape changed after shuffling weights"

    print("Random shuffle weight test passed.")

# Run the test
test_random_shuffle_weight()
