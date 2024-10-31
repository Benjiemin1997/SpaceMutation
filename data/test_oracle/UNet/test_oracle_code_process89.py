import unittest
import torch

from models.UNet.model_unet import UNet
from mut.neuron_effect_blocks import neuron_effect_block


class TestNeuronEffectBlock(unittest.TestCase):

    def setUp(self):
        self.model = UNet()



    def test_neuron_effect_block(self):
        neuron_effect_block(self.model, proportion=0.1)

        # Check if the model's parameters have been modified
        for name, param in self.model.named_parameters():
            if 'weight' in name:
                if 'linear_layers' in name or 'conv_layers' in name:
                    # Check if some weights have been set to zero
                    self.assertTrue(torch.any(param.data == 0))

        # Check if the model still works after mutation
        input_data = torch.randn(1, 1, 224, 224)
        original_output = self.model(input_data)
        mutated_model = neuron_effect_block(self.model, proportion=0.1)
        mutated_output = mutated_model(input_data)
        self.assertTrue(torch.allclose(original_output, mutated_output))

if __name__ == '__main__':
    unittest.main()

