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
            out = torch.relu(out)
            out = self.linear2(out)
            return out

    mock_model = MockModel()


    mutated_model = random_shuffle_weight(mock_model)


    expected_model_structure = """
    (linear1): Linear(in_features=10, out_features=10, bias=True)
    (relu): ReLU()
    (linear2): Linear(in_features=10, out_features=10, bias=True)
    """

    # Check the model's structure after mutation
    actual_model_structure = str(mutated_model)
    assert expected_model_structure in actual_model_structure, "Model structure does not match expected behavior."

    # Check that the weights have been shuffled
    original_linear1_weights = mock_model.linear1.weight.data.tolist()
    mutated_linear1_weights = mutated_model.linear1.weight.data.tolist()
    assert not torch.allclose(original_linear1_weights, mutated_linear1_weights), "Weights were not shuffled."

    print("Random shuffle weight test passed.")

if __name__ == "__main__":
    test_random_shuffle_weight()
