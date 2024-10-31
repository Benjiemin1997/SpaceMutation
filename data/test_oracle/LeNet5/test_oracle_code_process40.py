import unittest

import torch

from models.LeNet5.model_lenet5 import LeNet5
from mut.guassian_fuzz_splayers import gaussian_fuzzing_splayer
from mut.neuron_effect_blocks import neuron_effect_block


class TestNeuronEffectBlock(unittest.TestCase):

    def setUp(self):
        self.model = LeNet5()
        self.model.eval()

    def test_neuron_effect_block(self):

        original_model = self.model.state_dict().copy()

        # Apply neuron_effect_block with different proportions
        for proportion in [0.1, 0.5]:
            modified_model = neuron_effect_block(self.model, proportion)

            # Verify that the original model's state dict is unchanged
            self.assertEqual(original_model, self.model.state_dict())

            # Verify that some weights have been set to zero
            for name, param in modified_model.named_parameters():
                if 'weight' in name:
                    zeros = torch.eq(param.data, torch.zeros_like(param.data)).sum()
                    self.assertGreater(zeros, 0)


        gaussian_fuzzing_splayer(self.model)
        modified_model = neuron_effect_block(self.model, 0.1)
        self.assertNotEqual(modified_model.state_dict(), original_model)


if __name__ == '__main__':
    unittest.main()
