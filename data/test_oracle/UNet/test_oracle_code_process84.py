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

    original_state = {name: param.clone() for name, param in model.named_parameters()}
    print("Original State:", original_state)


    mutated_model = random_shuffle_weight(model)


    mutated_state = {name: param.clone() for name, param in mutated_model.named_parameters()}
    print("Mutated State:", mutated_state)


    for name, param in mutated_state.items():
        if name in original_state:
            assert not torch.allclose(original_state[name], param), f"Parameter {name} did not change as expected."
        else:
            raise ValueError(f"Unexpected parameter {name} in mutated state.")
