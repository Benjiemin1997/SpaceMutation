import unittest

import torch
from torch import nn

from models.LeNet5.model_lenet5 import LeNet5
from mut.guassian_fuzz_splayers import gaussian_fuzzing_splayer
from mut.neuron_effect_blocks import neuron_effect_block
from mut.random_shuffle import random_shuffle_weight
from mut.remove_activation import remove_activations
from mut.replace_activation import replace_activations
from mut.uniform_fuzz import uniform_fuzz_weight

class TestNeuronEffectBlock(unittest.TestCase):

    def setUp(self):
        self.model = LeNet5()
        self.model.eval()

    def test_neuron_effect_block(self):
        # Test case: Check if the model's weights are modified correctly by neuron_effect_block
        # Create a copy of the original model for comparison
        original_model = self.model.state_dict().copy()

        # Apply neuron_effect_block with different proportions
        for proportion in [0.1, 0.5]:
            mutated_model = neuron_effect_block(self.model, proportion)

            # Verify that the model's state_dict has been altered
            for layer in mutated_model.modules():
                if isinstance(layer, (nn.Linear, nn.Conv2d)):
                    original_weights = original_model[layer.__class__.__name__ + '.weight'].clone()
                    mutated_weights = mutated_model.state_dict()[layer.__class__.__name__ + '.weight']
                    self.assertFalse(torch.equal(original_weights, mutated_weights))

        # Test cases for mutation techniques
        # Gaussian fuzzing
        mutated_model_gaussian = gaussian_fuzzing_splayer(self.model)
        self.assertNotEqual(self.model.state_dict(), mutated_model_gaussian.state_dict())

        # Random shuffle
        mutated_model_random_shuffle = random_shuffle_weight(self.model)
        self.assertNotEqual(self.model.state_dict(), mutated_model_random_shuffle.state_dict())

        # Remove activations
        mutated_model_remove_activations = remove_activations(self.model)
        self.assertNotEqual(self.model.state_dict(), mutated_model_remove_activations.state_dict())

        # Replace activations
        mutated_model_replace_activations = replace_activations(self.model)
        self.assertNotEqual(self.model.state_dict(), mutated_model_replace_activations.state_dict())

        # Uniform fuzzing
        mutated_model_uniform_fuzz = uniform_fuzz_weight(self.model)
        self.assertNotEqual(self.model.state_dict(), mutated_model_uniform_fuzz.state_dict())

if __name__ == '__main__':
    unittest.main()
