import torch
from torch import nn
from mut.guassian_fuzz_splayers import gaussian_fuzzing_splayer
from mut.random_shuffle import random_shuffle_weight
from mut.remove_activation import remove_activations
from mut.replace_activation import replace_activations
from mut.uniform_fuzz import uniform_fuzz_weight

class TestModel(nn.Module):
    def __init__(self):
        super(TestModel, self).__init__()
        self.linear1 = nn.Linear(10, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        return x


def test_gaussian_fuzzing():
    model = TestModel()
    mutated_model = gaussian_fuzzing_splayer(model)

    for name, param in mutated_model.named_parameters():
        if 'weight' in name:
            assert torch.std(param) != 0, f"No standard deviation found in {name}"

def test_random_shuffle():
    model = TestModel()
    mutated_model = random_shuffle_weight(model)
    # Assert that parameters have been shuffled
    for name, param in model.named_parameters():
        if param is not mutated_model.get_parameter(name):
            assert not torch.equal(param, mutated_model.get_parameter(name)), f"Parameters in {name} were not shuffled"

def test_remove_activations():
    model = TestModel()
    mutated_model = remove_activations(model)

    for module in mutated_model.modules():
        assert not isinstance(module, nn.ReLU), f"Activation function found in {module}"

def test_replace_activations():
    model = TestModel()
    mutated_model = replace_activations(model, nn.Sigmoid())

    for module in mutated_model.modules():
        if isinstance(module, nn.ReLU):
            assert isinstance(mutated_model.replace_module(module, nn.Sigmoid()), nn.Sigmoid)

def test_uniform_fuzz():
    model = TestModel()
    mutated_model = uniform_fuzz_weight(model)
    for name, param in mutated_model.named_parameters():
        if 'weight' in name:
            assert torch.std(param) != 0, f"No standard deviation found in {name}"

# Run tests
if __name__ == "__main__":
    test_gaussian_fuzzing()
    test_random_shuffle()
    test_remove_activations()
    test_replace_activations()
    test_uniform_fuzz()
    print("All tests passed!")
