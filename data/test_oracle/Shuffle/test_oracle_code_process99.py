import unittest
import torch

from models.Shufflenetv2.shufflenetv2 import ShuffleNetV2
from mut.guassian_fuzz_splayers import gaussian_fuzzing_splayer
from mut.neuron_effect_blocks import neuron_effect_block
from mut.random_shuffle import random_shuffle_weight
from mut.remove_activation import remove_activations
from mut.replace_activation import replace_activations
from mut.uniform_fuzz import uniform_fuzz_weight

class TestNeuronEffectBlock(unittest.TestCase):

    def setUp(self):
        self.model = ShuffleNetV2()

    def test_neuron_effect_block(self):

        mutated_model = neuron_effect_block(self.model, proportion=0.1)
        mutated_weights = gaussian_fuzzing_splayer(mutated_model)
        self.assertNotEqual(mutated_model.fc.weight.sum(), 0, "Weights should be mutated after Gaussian fuzzing")

        mutated_model = neuron_effect_block(self.model, proportion=0.1)
        mutated_model = random_shuffle_weight(mutated_model)
        self.assertNotEqual(mutated_model.fc.weight.sum(), 0, "Weights should be mutated after random shuffling")

        # Remove Activations
        mutated_model = neuron_effect_block(self.model, proportion=0.1)
        mutated_model = remove_activations(mutated_model)
        self.assertIsNone(mutated_model.classifier[1], "Activation should be removed from the classifier")

        # Replace Activations
        mutated_model = neuron_effect_block(self.model, proportion=0.1)
        mutated_model = replace_activations(mutated_model)
        self.assertIsNotNone(mutated_model.classifier[1], "Activation should be replaced in the classifier")

        # Uniform Fuzzing
        mutated_model = neuron_effect_block(self.model, proportion=0.1)
        mutated_weights = uniform_fuzz_weight(mutated_model)
        self.assertNotEqual(mutated_model.fc.weight.sum(), 0, "Weights should be mutated after uniform fuzzing")

if __name__ == '__main__':
    unittest.main()