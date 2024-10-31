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
        shuffled_model = random_shuffle_weight(mutated_model)
        assert not torch.equal(shuffled_model.linear.weight, mutated_model.linear.weight), "Weights should be shuffled"

        # Remove activations test
        no_activ_model = remove_activations(shuffled_model)
        assert any([isinstance(module, nn.ReLU) or isinstance(module, nn.LeakyReLU) for name, module in no_activ_model.named_modules()]), "Activations should be removed"

        # Replace activations test
        replaced_model = replace_activations(no_activ_model)
        assert all([isinstance(module, nn.Sigmoid) for name, module in replaced_model.named_modules()]), "All activations should be replaced to Sigmoid"

        # Uniform fuzz test
        fuzzed_model = uniform_fuzz_weight(replaced_model)
        assert not torch.allclose(replaced_model.linear.weight, fuzzed_model.linear.weight), "Weights should be fuzzed"
