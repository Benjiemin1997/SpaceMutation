import unittest
import torch
import torch.nn as nn

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
        mutated_weights = gaussian_fuzzing_splayer(mutated_model)
        self.assertTrue(all(torch.allclose(mutated_weights[i], mutated_weights[i].clone(), atol=1e-5) for i in range(len(mutated_weights))))

        # Random Shuffle Weights
        mutated_model = neuron_effect_block(self.model, proportion=0.1)
        mutated_model = random_shuffle_weight(mutated_model)
        self.assertFalse(torch.equal(mutated_model.fc.weight, mutated_model.fc.weight.clone()))

        # Remove Activations
        mutated_model = neuron_effect_block(self.model, proportion=0.1)
        mutated_model = remove_activations(mutated_model)
        self.assertTrue(all(hasattr(layer, 'forward') for layer in mutated_model.modules()))

        # Replace Activations
        mutated_model = neuron_effect_block(self.model, proportion=0.1)
        mutated_model = replace_activations(mutated_model)
        self.assertTrue(all(isinstance(layer, nn.ReLU) for layer in mutated_model.modules()))

        # Uniform Fuzzing
        mutated_model = neuron_effect_block(self.model, proportion=0.1)
        mutated_weights = uniform_fuzz_weight(mutated_model)
        self.assertTrue(all(torch.allclose(mutated_weights[i], mutated_weights[i].clone(), atol=1e-5) for i in range(len(mutated_weights))))

if __name__ == '__main__':
    unittest.main()
