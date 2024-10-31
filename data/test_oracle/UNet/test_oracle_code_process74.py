import torch

from mut.random_shuffle import random_shuffle_weight


# Test Oracle Code

def test_random_shuffle_weight():
    # Create a mock model to test
    class MockModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = torch.nn.Linear(10, 10)
            self.linear2 = torch.nn.Linear(10, 10)

        def forward(self, x):
            out = self.linear1(x)
            out = self.linear2(out)
            return out

    # Create an instance of the mock model
    model = MockModel()

    # Perform random shuffle on the model's weights
    mutated_model = random_shuffle_weight(model)

    # Check that the number of parameters is the same before and after mutation
    assert model.numel() == mutated_model.numel(), "Number of parameters changed unexpectedly"

    # Check that the weights have been shuffled
    for param, mutated_param in zip(model.parameters(), mutated_model.parameters()):
        if torch.equal(param, mutated_param):
            assert False, "Weights were not shuffled as expected"
    
    print("Random shuffle weight test passed successfully.")
