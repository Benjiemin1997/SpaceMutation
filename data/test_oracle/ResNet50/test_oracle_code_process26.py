import torch
from torch import nn

from mut.reverse_activation import reverse_activations


def test_reverse_activations():
    class MockModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = nn.Linear(10, 5)
            self.relu = nn.ReLU()

        def forward(self, x):
            x = self.linear1(x)
            x = self.relu(x)
            return x

    model = MockModel()

    reversed_model = reverse_activations(model)

    input_data = torch.randn(1, 10)
    original_output = model(input_data)
    reversed_output = reversed_model(input_data)
    torch.testing.assert_allclose(original_output, -reversed_output)

    print("Reverse activation test passed.")

