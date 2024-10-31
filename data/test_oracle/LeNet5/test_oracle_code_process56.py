import unittest

import torch

from models.LeNet5.model_lenet5 import LeNet5
from mut.neuron_effect_blocks import neuron_effect_block


class TestNeuronEffectBlock(unittest.TestCase):

    def setUp(self):
        self.model = LeNet5()
        self.model.eval()

    def test_neuron_effect_block(self):
        # Test case: Check that the model's weights have been modified
        original_model_weights = self.model.state_dict()
        
        # Mutate the model using neuron_effect_block
        mutated_model = neuron_effect_block(self.model)
        

        mutated_model_weights = mutated_model.state_dict()
        for key in original_model_weights:
            if torch.equal(original_model_weights[key], mutated_model_weights[key]):
                self.fail(f"Weights for key {key} did not change as expected.")
        

        self.model.load_state_dict(original_model_weights)
        self.model.eval()

if __name__ == '__main__':
    unittest.main()
