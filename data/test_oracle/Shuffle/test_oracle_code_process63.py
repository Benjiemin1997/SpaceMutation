import torch

from models.Shufflenetv2.shufflenetv2 import ShuffleNetV2
from mut.random_shuffle import random_shuffle_weight


# Function to test the 'random_shuffle_weight' method
def test_random_shuffle_weight():
    # Create a mock model
    model = ShuffleNetV2()

    # Capture the original state of the model's parameters
    original_parameters = {param.name: param.data.clone() for param in model.parameters()}

    # Apply the 'random_shuffle_weight' method
    new_model = random_shuffle_weight(model)


    assert len(original_parameters) == len(new_model.state_dict()), "Number of parameters changed after shuffling"

    # Check that each parameter has been shuffled
    for name, original_param in original_parameters.items():
        shuffled_param = new_model.state_dict()[name]
        assert not torch.allclose(original_param, shuffled_param), f"Parameter '{name}' has not been shuffled"


    assert type(model) == type(new_model), "Model structure has changed after shuffling"
    
    # Clean up by restoring the original model's parameters
    for name, param in new_model.state_dict().items():
        new_model.state_dict()[name].copy_(original_parameters[name])

# Run the test
test_random_shuffle_weight()