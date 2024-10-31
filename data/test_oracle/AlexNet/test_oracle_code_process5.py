import torch
from torch import nn
from mut.guassian_fuzz_splayers import gaussian_fuzzing_splayer
from mut.random_shuffle import random_shuffle_weight
from mut.remove_activation import remove_activations
from mut.replace_activation import replace_activations

from mut.uniform_fuzz import uniform_fuzz_weight

class TestModelMutations:

    def test_gaussian_fuzzing_splayer(self):

        class SimpleModel(nn.Module):
            def __init__(self):
                super(SimpleModel, self).__init__()
                self.linear = nn.Linear(10, 10)

            def forward(self, x):
                return self.linear(x)

        model = SimpleModel()


        mutated_model = gaussian_fuzzing_splayer(model)


        assert not torch.allclose(model.linear.weight, mutated_model.linear.weight)

    def test_random_shuffle_weight(self):

        class SimpleModel(nn.Module):
            def __init__(self):
                super(SimpleModel, self).__init__()
                self.linear = nn.Linear(10, 10)

            def forward(self, x):
                return self.linear(x)

        model = SimpleModel()


        mutated_model = random_shuffle_weight(model)


        assert not torch.equal(model.linear.weight, mutated_model.linear.weight)

    def test_remove_activations(self):

        class SimpleModel(nn.Module):
            def __init__(self):
                super(SimpleModel, self).__init__()
                self.linear1 = nn.Linear(10, 10)
                self.relu = nn.ReLU()

            def forward(self, x):
                return self.relu(self.linear1(x))

        model = SimpleModel()

        mutated_model = remove_activations(model)


        assert 'relu' not in str(mutated_model)

    def test_replace_activations(self):

        class SimpleModel(nn.Module):
            def __init__(self):
                super(SimpleModel, self).__init__()
                self.linear1 = nn.Linear(10, 10)
                self.relu = nn.ReLU()

            def forward(self, x):
                return self.relu(self.linear1(x))

        model = SimpleModel()

        mutated_model = replace_activations(model)


        assert 'relu' not in str(mutated_model)
        assert 'sigmoid' in str(mutated_model)

    def test_uniform_fuzz_weight(self):
        class SimpleModel(nn.Module):
            def __init__(self):
                super(SimpleModel, self).__init__()
                self.linear = nn.Linear(10, 10)

            def forward(self, x):
                return self.linear(x)

        model = SimpleModel()


        mutated_model = uniform_fuzz_weight(model)

        assert not torch.allclose(model.linear.weight, mutated_model.linear.weight)
