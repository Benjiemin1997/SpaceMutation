import torch
from torch import nn

from mut.reverse_activation import reverse_activations


def test_reverse_activations():
    class MockModel(nn.Module):
        def __init__(self):
            super(MockModel, self).__init__()
            self.layer1 = nn.Sequential(nn.Linear(10, 5), nn.ReLU())
            self.layer2 = nn.Sequential(nn.Linear(5, 3), nn.Tanh())

    x = torch.randn(1, 10)
    original_model = MockModel()
    original_output = original_model(x)

    reversed_model = reverse_activations(original_model)
    reversed_output = reversed_model(x)

    assert torch.allclose(-original_output, reversed_output), "The reverse activations did not work as expected."
    assert isinstance(reversed_model.layer1[1], nn.ReLU), "The activation was not reversed correctly."
    assert isinstance(reversed_model.layer2[1], nn.Tanh), "The activation was not reversed correctly."
    
    print("All tests passed successfully.")
