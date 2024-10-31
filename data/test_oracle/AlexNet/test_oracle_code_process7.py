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
        self.proportion = 0.1

    def test_neuron_effect_block(self):
        mutated_model = neuron_effect_block(self.model, self.proportion)
        self.assertNotEqual(mutated_model, self.model)


        for name, module in mutated_model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                self.assertTrue(torch.sum(module.weight.data) < torch.sum(self.model.state_dict()[name]['weight'].data))


        self.fuzzing_tests(mutated_model)


        self.random_shuffling_tests(mutated_model)


        self.activation_removal_tests(mutated_model)


        self.activation_replacement_tests(mutated_model)

        self.uniform_fuzzing_tests(mutated_model)

    def fuzzing_tests(self, model):

        gaussian_fuzzing_splayer(model)
        self.assertNotEqual(model, self.model)


        uniform_fuzz_weight(model)
        self.assertNotEqual(model, self.model)

    def random_shuffling_tests(self, model):
        random_shuffle_weight(model)
        self.assertNotEqual(model, self.model)

    def activation_removal_tests(self, model):
        remove_activations(model)
        self.assertNotEqual(model, self.model)

    def activation_replacement_tests(self, model):
        replace_activations(model)
        self.assertNotEqual(model, self.model)

    def uniform_fuzzing_tests(self, model):
        uniform_fuzz_weight(model)
        self.assertNotEqual(model, self.model)

if __name__ == '__main__':
    unittest.main()