import unittest
import torch

from models.AlexNet.model_alexnet import AlexNet
from mut.guassian_fuzz_splayers import gaussian_fuzzing_splayer
from mut.neuron_effect_blocks import neuron_effect_block
from mut.random_shuffle import random_shuffle_weight
from mut.remove_activation import remove_activations
from mut.replace_activation import replace_activations
from mut.uniform_fuzz import uniform_fuzz_weight

class TestNeuronEffectBlock(unittest.TestCase):

    def setUp(self):
        self.model = AlexNet()

    def test_neuron_effect_block(self):
        # Gaussian Fuzzing
        mutated_model = neuron_effect_block(self.model, proportion=0.1)
        mutated_weights = gaussian_fuzzing_splayer(mutated_model, std_dev=0.1)
        self.assertNotEqual(mutated_model.fc.weight.sum(), 0, "Weights should be mutated after Gaussian fuzzing")

        # Random Shuffle Weights
        mutated_model = neuron_effect_block(self.model, proportion=0.1)
        mutated_model = random_shuffle_weight(mutated_model)
        self.assertNotEqual(mutated_model.fc.weight.sum(), 0, "Weights should be mutated after random shuffle")

        # Remove Activations
        mutated_model = neuron_effect_block(self.model, proportion=0.1)
        mutated_model = remove_activations(mutated_model)
        self.assertEqual(list(mutated_model.modules()), [], "Model should have no activations after removing all")

        # Replace Activations
        mutated_model = neuron_effect_block(self.model, proportion=0.1)
        mutated_model = replace_activations(mutated_model)
        self.assertEqual(list(mutated_model.modules()), [], "Model should have no activations after replacing all")

        # Uniform Fuzzing
        mutated_model = neuron_effect_block(self.model, proportion=0.1)
        mutated_weights = uniform_fuzz_weight(mutated_model)
        self.assertNotEqual(mutated_model.fc.weight.sum(), 0, "Weights should be mutated after uniform fuzzing")

if __name__ == '__main__':
    unittest.main()