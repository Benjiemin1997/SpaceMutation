import torch
from torch import nn

from models.LeNet5.model_lenet5 import LeNet5
from mut.guassian_fuzz_splayers import gaussian_fuzzing_splayer
from mut.random_shuffle import random_shuffle_weight
from mut.remove_activation import remove_activations
from mut.replace_activation import replace_activations
from mut.uniform_fuzz import uniform_fuzz_weight

class TestModelFunctions:

    def setup(self):
        self.model = LeNet5()

    def test_gaussian_fuzzing_splayer(self):
        # Test Gaussian Fuzzing Splayer Function
        mutated_model = gaussian_fuzzing_splayer(self.model, std_ratio=0.5)
        assert isinstance(mutated_model, nn.Module), "The output should be an instance of nn.Module"

        # Test for Linear Layer Modification
        for name, layer in mutated_model.named_modules():
            if isinstance(layer, nn.Linear):
                for param in layer.parameters():
                    assert torch.is_tensor(param.data), "Parameter data should be a tensor"
                    assert torch.std(param.data) != 0, "Standard deviation of parameter data should not be zero"

    def test_random_shuffle_weight(self):
        # Test Random Shuffle Weight Function
        shuffled_model = random_shuffle_weight(self.model)
        assert isinstance(shuffled_model, nn.Module), "The output should be an instance of nn.Module"
        for name, layer in shuffled_model.named_modules():
            if isinstance(layer, nn.Linear):
                assert not torch.equal(layer.weight, self.model.get_submodule(name).weight), "Weight should have been shuffled"

    def test_remove_activations(self):
        # Test Remove Activations Function
        no_activations_model = remove_activations(self.model)
        assert isinstance(no_activations_model, nn.Module), "The output should be an instance of nn.Module"
        assert 'GELU' not in str(no_activations_model), "GELU activation should have been removed from the model"

    def test_replace_activations(self):
        # Test Replace Activations Function
        replaced_activations_model = replace_activations(self.model)
        assert isinstance(replaced_activations_model, nn.Module), "The output should be an instance of nn.Module"
        assert 'ReLU' in str(replaced_activations_model), "ReLU activation should have replaced GELU activation"

    def test_uniform_fuzz_weight(self):
        # Test Uniform Fuzz Weight Function
        fuzzed_model = uniform_fuzz_weight(self.model)
        assert isinstance(fuzzed_model, nn.Module), "The output should be an instance of nn.Module"
        for name, layer in fuzzed_model.named_modules():
            if isinstance(layer, nn.Linear):
                assert torch.allclose(layer.weight, self.model.get_submodule(name).weight + torch.rand_like(layer.weight) * 0.1, atol=1e-2), "Weights should have been fuzzed uniformly"

if __name__ == "__main__":
    test = TestModelFunctions()
    test.setup()
    test.test_gaussian_fuzzing_splayer()
    test.test_random_shuffle_weight()
    test.test_remove_activations()
    test.test_replace_activations()
    test.test_uniform_fuzz_weight()
