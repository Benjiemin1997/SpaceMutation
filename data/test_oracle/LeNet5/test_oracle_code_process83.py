import unittest

import torch
from torch import nn

from models.LeNet5.model_lenet5 import LeNet5
from mut.neuron_effect_blocks import neuron_effect_block


class TestNeuronEffectBlock(unittest.TestCase):

    def setUp(self):
        self.model = LeNet5()
        self.model.eval()

    def test_neuron_effect_block(self):
        # Test case: Check if the model's weights have been modified after applying neuron_effect_block
        neuron_effect_block(self.model)

        # Check if any of the Linear or Conv2d layers' weights have been altered
        for module in self.model.modules():
            if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
                for weight in module.parameters():
                    if torch.all(weight == 0):
                        break
                else:
                    # If no zero-weight is found in this module, fail the test
                    self.fail(f"No zero-weight found in {module}")

        # Optional: Check if the expected exceptions are raised
        with self.assertRaises(ValueError):
            neuron_effect_block(None)

        # Optional: Additional checks can be added here, depending on specific requirements

if __name__ == '__main__':
    unittest.main()
