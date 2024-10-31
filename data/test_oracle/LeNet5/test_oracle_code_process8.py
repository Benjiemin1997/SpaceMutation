import torch

from mut.random_shuffle import random_shuffle_weight


# Test Oracle Code

def test_random_shuffle_weight():
    # Initialize a simple model (for demonstration purposes)
    class SimpleModel(torch.nn.Module):
        def __init__(self):
            super(SimpleModel, self).__init__()
            self.linear = torch.nn.Linear(10, 5)

        def forward(self, x):
            return self.linear(x)

    # Create an instance of the model
    model = SimpleModel()

    # Apply random shuffle to the model's weights
    model = random_shuffle_weight(model)

    # Define expected behavior or conditions after shuffling
    # For simplicity, we assume that the output changes significantly
    input_data = torch.randn(1, 10)
    original_output = model(input_data)
    shuffled_output = model(input_data)

    # Check if the output changes significantly after shuffling
    assert not torch.allclose(original_output, shuffled_output, atol=1e-4), "The output did not change significantly after shuffling the weights."

    print("Random Shuffle Weight Test Passed")

if __name__ == "__main__":
    test_random_shuffle_weight()
