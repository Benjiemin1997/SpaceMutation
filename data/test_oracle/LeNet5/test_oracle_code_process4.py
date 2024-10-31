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
            assert torch.is_tensor(param)
            assert torch.std(param) > 0.5 * torch.std(self.model.state_dict()[name])
        
    def test_random_shuffle_weight(self):
        # Test Random Shuffle Weight Function
        shuffled_model = random_shuffle_weight(self.model)
        for name, param in shuffled_model.named_parameters():
            assert not torch.equal(param, self.model.state_dict()[name])

    def test_remove_activations(self):
        # Test Remove Activations Function
        no_act_model = remove_activations(self.model)
        assert len(list(no_act_model.children())) == 1
        
    def test_replace_activations(self):
        # Test Replace Activations Function
        replaced_model = replace_activations(self.model)
        for name, module in replaced_model.named_modules():
            if 'ReLU' in str(type(module)):
                assert isinstance(module, nn.ReLU)

    def test_uniform_fuzz_weight(self):
        # Test Uniform Fuzz Weight Function
        fuzzed_model = uniform_fuzz_weight(self.model)
        for name, param in fuzzed_model.named_parameters():
            assert torch.abs(torch.mean(param)) > 0.1 * torch.std(self.model.state_dict()[name])
