import unittest
import torch
import torch.nn as nn

from models.UNet.model_unet import UNet
from mut.neuron_effect_blocks import neuron_effect_block


class TestNeuronEffectBlock(unittest.TestCase):

    def setUp(self):
        self.model = UNet()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_neuron_effect_block(self):
        neuron_effect_block(self.model, proportion=0.1)
        

        for module in self.model.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                original_weights = module.weight.clone().detach().to(self.device)
                module.to(self.device)
                neuron_effect_block(module, proportion=0.1)
                
                # Check if any weights have been set to zero
                self.assertTrue(torch.any(original_weights == 0) and torch.any(module.weight == 0),
                                msg="Weights were not set to zero as expected.")
                break
                

        self.assertNotEqual(self.model.state_dict(), self.model.state_dict(), msg="State dict did not change after applying neuron_effect_block.")

if __name__ == '__main__':
    unittest.main()
