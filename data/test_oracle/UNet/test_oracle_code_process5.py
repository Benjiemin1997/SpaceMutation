import unittest
import torch
import numpy as np
from torch import nn

from models.UNet.model_unet import UNet
from mut.guassian_fuzz_splayers import gaussian_fuzzing_splayer
from mut.random_shuffle import random_shuffle_weight
from mut.remove_activation import remove_activations
from mut.replace_activation import replace_activations

from mut.uniform_fuzz import uniform_fuzz_weight

class TestMutations(unittest.TestCase):

    def setUp(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = UNet().to(self.device)
        self.input_data = torch.randn(1, 1, 64, 64).to(self.device)

    def test_gaussian_fuzzing_splayer(self):
        mutated_model = gaussian_fuzzing_splayer(self.model, std_ratio=0.5)
        for name, layer in mutated_model.named_modules():
            if isinstance(layer, nn.Linear):
                weights = layer.weight.data.cpu().numpy()
                std = np.std(weights)
                fuzzed_std = std * 0.5
                fuzzed_weights = weights + np.random.normal(0, fuzzed_std, weights.shape)
                self.assertTrue(np.allclose(layer.weight.data.cpu().numpy(), fuzzed_weights))

    def test_random_shuffle_weight(self):
        mutated_model = random_shuffle_weight(self.model)
        for name, layer in mutated_model.named_modules():
            if isinstance(layer, nn.Linear):
                shuffled_weights = layer.weight.data.cpu().numpy()
                original_weights = layer.weight.data.cpu().numpy()
                self.assertFalse(np.allclose(shuffled_weights, original_weights))

    def test_remove_activations(self):
        mutated_model = remove_activations(self.model)
        for name, layer in mutated_model.named_modules():
            if isinstance(layer, nn.ReLU):
                self.assertIsNone(layer.activation)

    def test_replace_activations(self):
        mutated_model = replace_activations(self.model)
        for name, layer in mutated_model.named_modules():
            if isinstance(layer, nn.ReLU):
                self.assertIsInstance(layer, nn.LeakyReLU)

    def test_uniform_fuzz_weight(self):
        mutated_model = uniform_fuzz_weight(self.model)
        for name, layer in mutated_model.named_modules():
            if isinstance(layer, nn.Linear):
                weights = layer.weight.data.cpu().numpy()
                fuzzed_weights = weights + np.random.uniform(-0.1, 0.1, weights.shape)
                self.assertTrue(np.allclose(layer.weight.data.cpu().numpy(), fuzzed_weights))

if __name__ == '__main__':
    unittest.main()
