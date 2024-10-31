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
        self.proportion = 0.1

    def test_neuron_effect_block(self):
        mutated_model = neuron_effect_block(self.model, self.proportion)
        self.assertNotEqual(mutated_model, self.model)

        # Check if any weights have been modified
        for name, param in mutated_model.named_parameters():
            if 'weight' in name:
                self.assertTrue(torch.sum(param) != 0)

        # Check if the model still runs without errors
        dummy_input = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            output = mutated_model(dummy_input)
        self.assertIsNotNone(output)

        # Check for additional mutations using other mutation techniques
        mutated_model = gaussian_fuzzing_splayer(mutated_model)
        self.assertNotEqual(mutated_model, self.model)

        mutated_model = random_shuffle_weight(mutated_model)
        self.assertNotEqual(mutated_model, self.model)

        mutated_model = remove_activations(mutated_model)
        self.assertNotEqual(mutated_model, self.model)

        mutated_model = replace_activations(mutated_model)
        self.assertNotEqual(mutated_model, self.model)

        mutated_model = uniform_fuzz_weight(mutated_model)
        self.assertNotEqual(mutated_model, self.model)

if __name__ == '__main__':
    unittest.main()