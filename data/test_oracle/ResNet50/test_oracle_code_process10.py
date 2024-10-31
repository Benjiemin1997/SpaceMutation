import torch
from torch import nn

from mut.reverse_activation import reverse_activations


def test_reverse_activations():
    class MockModel(nn.Module):
        def __init__(self):
            super(MockModel, self).__init__()
            self.layer1 = nn.Sequential(nn.Linear(10, 5), nn.ReLU())
            self.layer2 = nn.Sequential(nn.Linear(5, 3), nn.Tanh())
    

    input_data = torch.randn(1, 10)

    original_output = MockModel()(input_data)

    reversed_model = reverse_activations(MockModel())
    reversed_output = reversed_model(input_data)

    assert not isinstance(reversed_model.layer1[1], nn.ReLU), "ReLU was not reversed."
    assert isinstance(reversed_model.layer1[1], nn.Identity), "Layer 1 activation should be an Identity layer."
    assert not isinstance(reversed_model.layer2[1], nn.Tanh), "Tanh was not reversed."
    assert isinstance(reversed_model.layer2[1], nn.Identity), "Layer 2 activation should be an Identity layer."
    assert torch.allclose(original_output.abs(), reversed_output.abs()), "Output magnitudes are not consistent after reversing activations."

if __name__ == "__main__":
    test_reverse_activations()
