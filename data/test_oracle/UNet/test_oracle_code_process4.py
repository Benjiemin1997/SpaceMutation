import unittest

from models.UNet.model_unet import UNet
from mut.guassian_fuzz_splayers import gaussian_fuzzing_splayer
from mut.neuron_effect_blocks import neuron_effect_block
from mut.random_shuffle import random_shuffle_weight
from mut.remove_activation import remove_activations
from mut.replace_activation import replace_activations
from mut.uniform_fuzz import uniform_fuzz_weight

class TestNeuronEffectBlock(unittest.TestCase):

    def setUp(self):
        self.model = UNet()

    def test_neuron_effect_block(self):
        original_state_dict = self.model.state_dict()

        # Apply neuron effect block with proportion 0.1
        mutated_model = neuron_effect_block(self.model, proportion=0.1)

        # Check if model state has been mutated
        mutated_state_dict = mutated_model.state_dict()
        for key in original_state_dict:
            if key in mutated_state_dict:
                self.assertNotEqual(original_state_dict[key], mutated_state_dict[key])

        # Apply various mutation techniques on the mutated model
        gaussian_fuzzing_splayer(mutated_model)
        random_shuffle_weight(mutated_model)
        remove_activations(mutated_model)
        replace_activations(mutated_model)
        uniform_fuzz_weight(mutated_model)



if __name__ == '__main__':
    unittest.main()
