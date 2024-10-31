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
        assert isinstance(mutated_model, nn.Module), "The returned model should be an instance of nn.Module"
        
        # Check if any of the layers have been perturbed
        for name, layer in mutated_model.named_modules():
            if isinstance(layer, nn.Linear):
                assert torch.is_tensor(layer.weight), "Linear layer weight should be a tensor"
                assert not torch.equal(layer.weight, self.model.get_submodule(name).weight), "Layer weights should not be equal"

    def test_random_shuffle_weight(self):
        # Test Random Shuffle Weight Function
        mutated_model = random_shuffle_weight(self.model)
        assert isinstance(mutated_model, nn.Module), "The returned model should be an instance of nn.Module"
        
        # Check if any of the layers' weights have been shuffled
        for name, layer in mutated_model.named_modules():
            if isinstance(layer, nn.Linear):
                assert not torch.equal(layer.weight, self.model.get_submodule(name).weight), "Layer weights should not be equal"

    def test_remove_activations(self):
        # Test Remove Activations Function
        mutated_model = remove_activations(self.model)
        assert isinstance(mutated_model, nn.Module), "The returned model should be an instance of nn.Module"
        
        # Check if any activations have been removed
        for name, module in mutated_model.named_modules():
            assert not isinstance(module, nn.ReLU), "Should not find ReLU activation in the mutated model"

    def test_replace_activations(self):
        # Test Replace Activations Function
        mutated_model = replace_activations(self.model)
        assert isinstance(mutated_model, nn.Module), "The returned model should be an instance of nn.Module"
        
        # Check if all ReLU activations have been replaced by Sigmoid
        for name, module in mutated_model.named_modules():
            if isinstance(module, nn.ReLU):
                assert isinstance(self.model.get_submodule(name), nn.ReLU), "Original ReLU should exist in the original model"
                assert isinstance(module, nn.Sigmoid), "Replaced ReLU should be replaced by Sigmoid"

    def test_uniform_fuzz_weight(self):
        # Test Uniform Fuzz Weight Function
        mutated_model = uniform_fuzz_weight(self.model)
        assert isinstance(mutated_model, nn.Module), "The returned model should be an instance of nn.Module"
        
        # Check if any of the layers' weights have been fuzzed uniformly
        for name, layer in mutated_model.named_modules():
            if isinstance(layer, nn.Linear):
                assert torch.is_tensor(layer.weight), "Linear layer weight should be a tensor"
                assert not torch.equal(layer.weight, self.model.get_submodule(name).weight), "Layer weights should not be equal"
                assert torch.min(layer.weight) != torch.max(layer.weight), "Uniform fuzzing should result in non-constant weights"

