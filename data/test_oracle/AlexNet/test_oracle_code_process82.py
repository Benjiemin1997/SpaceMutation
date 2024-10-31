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
        self.proportion = 0.1

    def test_neuron_effect_block(self):
        mutated_model = neuron_effect_block(self.model, self.proportion)
        self.assertNotEqual(mutated_model, self.model)

        # Check if any of the mutated layers have been altered
        for name, module in mutated_model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                self.assertTrue(torch.sum(module.weight.data) < torch.sum(self.model.state_dict()[name]['weight'].data))

        # Fuzzing tests
        self.fuzzing_tests(mutated_model)

        # Random shuffle tests
        self.random_shuffling_tests(mutated_model)

        # Remove activation tests
        self.activation_removal_tests(mutated_model)

        # Replace activation tests
        self.activation_replacement_tests(mutated_model)

        # Uniform fuzzing tests
        self.uniform_fuzzing_tests(mutated_model)

    def fuzzing_tests(self, model):
        # Apply Gaussian fuzzing
        gaussian_fuzzing_splayer(model)
        
        # Apply uniform fuzzing
        uniform_fuzz_weight(model)

        # Ensure the model is still in an expected state after fuzzing
        self.assertTrue(torch.allclose(model.state_dict()['features.0.weight'], self.model.state_dict()['features.0.weight']))
        self.assertTrue(torch.allclose(model.state_dict()['features.3.weight'], self.model.state_dict()['features.3.weight']))
        self.assertTrue(torch.allclose(model.state_dict()['classifier.1.weight'], self.model.state_dict()['classifier.1.weight']))

    def random_shuffling_tests(self, model):
        random_shuffle_weight(model)
        self.assertFalse(torch.allclose(model.state_dict()['features.0.weight'], self.model.state_dict()['features.0.weight']))
        self.assertFalse(torch.allclose(model.state_dict()['features.3.weight'], self.model.state_dict()['features.3.weight']))
        self.assertFalse(torch.allclose(model.state_dict()['classifier.1.weight'], self.model.state_dict()['classifier.1.weight']))

    def activation_removal_tests(self, model):
        remove_activations(model)
        self.assertIsNone(model.state_dict().get('features.4.bias'))

    def activation_replacement_tests(self, model):
        replace_activations(model)
        self.assertEqual(model.state_dict()['features.4.bias'].item(), 0)

    def uniform_fuzzing_tests(self, model):
        uniform_fuzz_weight(model)
        self.assertNotEqual(model.state_dict()['features.0.weight'], self.model.state_dict()['features.0.weight'])
        self.assertNotEqual(model.state_dict()['features.3.weight'], self.model.state_dict()['features.3.weight'])
        self.assertNotEqual(model.state_dict()['classifier.1.weight'], self.model.state_dict()['classifier.1.weight'])

if __name__ == '__main__':
    unittest.main()
