import unittest
import torch
import torch.nn as nn

from models.UNet.model_unet import UNet
from mut.guassian_fuzz_splayers import gaussian_fuzzing_splayer
from mut.random_shuffle import random_shuffle_weight
from mut.remove_activation import remove_activations
from mut.replace_activation import replace_activations

from mut.uniform_fuzz import uniform_fuzz_weight

class TestModelMutations(unittest.TestCase):

    def setUp(self):
        self.model = UNet()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def test_gaussian_fuzzing(self):
        model = gaussian_fuzzing_splayer(self.model, std_ratio=0.5)

        for name, param in model.named_parameters():
            if 'weight' in name:
                self.assertTrue(torch.std(param).item() > 0.5)

    def test_random_shuffle(self):
        model = random_shuffle_weight(self.model)
        # Add assertions to check that the model's weights have been shuffled
        for name, param in model.named_parameters():
            if 'weight' in name:
                self.assertFalse(torch.equal(param, self.model.state_dict()[name]))

    def test_remove_activations(self):
        model = remove_activations(self.model)

        for name, module in model.named_modules():
            if isinstance(module, nn.ReLU):
                self.assertTrue(isinstance(module, nn.Identity))

    def test_replace_activations(self):
        model = replace_activations(self.model)

        for name, module in model.named_modules():
            if isinstance(module, nn.ReLU):
                self.assertTrue(isinstance(module, nn.SiLU))

    def test_uniform_fuzz(self):
        model = uniform_fuzz_weight(self.model, std_ratio=0.1)
        # Add assertions to check that the model's parameters have been uniformly fuzzed
        for name, param in model.named_parameters():
            if 'weight' in name:
                self.assertTrue(torch.min(param).item() >= -0.1)
                self.assertTrue(torch.max(param).item() <= 0.1)

if __name__ == '__main__':
    unittest.main()
