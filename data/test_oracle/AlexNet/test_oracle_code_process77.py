import unittest
import torch
import torch.nn as nn

from mut.guassian_fuzz_splayers import gaussian_fuzzing_splayer
from mut.neuron_effect_blocks import neuron_effect_block
from mut.random_shuffle import random_shuffle_weight
from mut.remove_activation import remove_activations
from mut.replace_activation import replace_activations
from mut.uniform_fuzz import uniform_fuzz_weight

class TestNeuronEffectBlock(unittest.TestCase):

    def setUp(self):
        self.model = torch.hub.load('pytorch/vision:v0.10.0', 'alexnet', pretrained=True).eval()

    def test_neuron_effect_block(self):
        # Gaussian Fuzzing
        mutated_model = neuron_effect_block(self.model, proportion=0.1)
        mutated_weights = gaussian_fuzzing_splayer(mutated_model)
        self.assertNotEqual(mutated_model.fc.weight.sum(), 0, "Weights should be mutated.")
        
        # Random Shuffling
        mutated_model = neuron_effect_block(self.model, proportion=0.1)
        mutated_weights = random_shuffle_weight(mutated_model)
        self.assertNotEqual(mutated_model.fc.weight.sum(), 0, "Weights should be mutated.")
        
        # Remove Activations
        mutated_model = neuron_effect_block(self.model, proportion=0.1)
        mutated_model = remove_activations(mutated_model)
        self.assertEqual(len(list(mutated_model.modules())), len(list(self.model.modules())), "Number of modules should remain unchanged.")
        
        # Replace Activations
        mutated_model = neuron_effect_block(self.model, proportion=0.1)
        mutated_model = replace_activations(mutated_model, nn.ReLU())
        self.assertIsInstance(mutated_model.fc.activation, nn.ReLU, "Activation function should be replaced.")
        
        # Uniform Fuzzing
        mutated_model = neuron_effect_block(self.model, proportion=0.1)
        mutated_weights = uniform_fuzz_weight(mutated_model)
        self.assertNotEqual(mutated_model.fc.weight.sum(), 0, "Weights should be mutated.")

if __name__ == '__main__':
    unittest.main()
