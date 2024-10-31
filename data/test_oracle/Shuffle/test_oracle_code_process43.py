import torch

from models.Shufflenetv2.shufflenetv2 import ShuffleNetV2
from mut.random_shuffle import random_shuffle_weight


# Function to test the 'random_shuffle_weight' method
def test_random_shuffle_weight():
    # Create a mock model
    model = ShuffleNetV2()

    # Capture the original state of the model's parameters
    original_params = {param.name: param.data.clone() for param in model.parameters()}

    # Apply the 'random_shuffle_weight' method
    new_model = random_shuffle_weight(model)

    # Assert that the number of parameters remains the same
    assert len(original_params) == len(new_model.state_dict()), "Number of parameters changed after shuffling"

    # Check that each parameter has been shuffled
    for name, param in new_model.named_parameters():
        assert not torch.allclose(param.data, original_params[name]), f"Parameter '{name}' did not change after shuffling"

    print("All tests passed successfully!")

# Run the test
test_random_shuffle_weight()
