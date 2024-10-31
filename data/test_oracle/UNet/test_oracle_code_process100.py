import torch

from mut.random_shuffle import random_shuffle_weight


# Test Oracle Code

def test_random_shuffle_weight():
    # Create a simple model to test
    class SimpleModel(torch.nn.Module):
        def __init__(self):
            super(SimpleModel, self).__init__()
            self.linear = torch.nn.Linear(10, 5)

        def forward(self, x):
            return self.linear(x)

    # Initialize the model
    model = SimpleModel()

    # Check original state
    original_state = {name: param.clone() for name, param in model.state_dict().items()}

    # Apply the random shuffle weight function
    model = random_shuffle_weight(model)

    # Check shuffled state
    shuffled_state = {name: param.clone() for name, param in model.state_dict().items()}
    assert not all(torch.equal(original, shuffled) for original, shuffled in
                   zip(original_state.values(), shuffled_state.values())), "Weights have not been shuffled."

    # Reset model to original state
    model.load_state_dict(original_state)

    # Additional checks to ensure functionality is as expected
    input_data = torch.randn(1, 10)
    output_before = model(input_data)
    model = random_shuffle_weight(model)
    output_after = model(input_data)
    assert not torch.allclose(output_before,
                              output_after), "The model output after shuffling weights should be different from before."

