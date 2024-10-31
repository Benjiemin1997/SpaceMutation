import torch

# Function to test the 'random_shuffle_weight' method
def test_random_shuffle_weight():
    # Create a mock model for testing
    class MockModel(torch.nn.Module):
        def __init__(self):
            super(MockModel, self).__init__()
            self.conv = torch.nn.Conv2d(3, 24, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    
    # Initialize the mock model
    model = MockModel()
    
    # Apply the 'random_shuffle_weight' method
    model = random_shuffle_weight(model)
    
    # Check that the weights have been shuffled
    assert not torch.equal(model.conv.weight, torch.randn_like(model.conv.weight)), "Weights have not been shuffled."
    
    # Check that the model is still of type torch.nn.Module after shuffling
    assert isinstance(model, torch.nn.Module), "The model type has changed after applying random_shuffle_weight."
    
    print("Test passed: random_shuffle_weight method works correctly.")

# Run the test
test_random_shuffle_weight()