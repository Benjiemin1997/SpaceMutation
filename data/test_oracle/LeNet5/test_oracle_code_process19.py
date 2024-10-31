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
        fuzzed_model = gaussian_fuzzing_splayer(self.model, std_ratio=0.5)
        for name, param in fuzzed_model.named_parameters():
            assert torch.is_tensor(param), f"Parameter {name} is not a tensor"


    def test_remove_activations(self):
        no_activations_model = remove_activations(self.model)
        for name, module in no_activations_model.named_modules():
            assert not hasattr(module, "activation"), f"Activation found in module {name}"

    def test_replace_activations(self):
        # Test Replace Activations Function
        replaced_activations_model = replace_activations(self.model)
        for name, module in replaced_activations_model.named_modules():
            if "activation" in name:
                assert isinstance(module, nn.ReLU), f"Activation type in module {name} is not ReLU"

    def test_uniform_fuzz_weight(self):
        # Test Uniform Fuzz Weight Function
        fuzzed_weights_model = uniform_fuzz_weight(self.model)
        for name, param in fuzzed_weights_model.named_parameters():
            assert torch.is_tensor(param), f"Parameter {name} is not a tensor"
            assert torch.allclose(param, uniform_fuzz_weight(param.data)), f"Parameter {name} is not uniformly fuzzed correctly"