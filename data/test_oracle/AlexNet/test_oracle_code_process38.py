import torch
from torch import nn
from mut.guassian_fuzz_splayers import gaussian_fuzzing_splayer
from mut.random_shuffle import random_shuffle_weight
from mut.remove_activation import remove_activations
from mut.replace_activation import replace_activations

from mut.uniform_fuzz import uniform_fuzz_weight

class TestModelMutations:

    def test_gaussian_fuzzing_splayer(self):
        # Create a simple model for testing
        class SimpleModel(nn.Module):
            def __init__(self):
                super(SimpleModel, self).__init__()
                self.linear = nn.Linear(10, 10)

            def forward(self, x):
                return self.linear(x)

        model = SimpleModel()

        # Fuzzing test
        mutated_model = gaussian_fuzzing_splayer(model, std_ratio=0.5)
        assert not torch.allclose(model.linear.weight, mutated_model.linear.weight), "Weights should be mutated"

        # Random shuffle test
        mutated_model = random_shuffle_weight(mutated_model)
        assert not torch.equal(model.linear.weight, mutated_model.linear.weight), "Weights should be shuffled"

        # Remove activations test
        mutated_model = remove_activations(mutated_model)
        assert len(mutated_model.children()) > 0, "Activations should not be removed"

        # Replace activations test
        mutated_model = replace_activations(mutated_model, nn.ReLU())
        assert isinstance(mutated_model.linear, nn.ReLU), "Activation should be replaced"

        # Uniform fuzz test
        mutated_model = uniform_fuzz_weight(mutated_model, std_ratio=0.5)
        assert not torch.allclose(model.linear.weight, mutated_model.linear.weight), "Weights should be mutated by uniform fuzz"

if __name__ == "__main__":
    tester = TestModelMutations()
    tester.test_gaussian_fuzzing_splayer()
