import unittest
import torch

from models.UNet.model_unet import UNet
from mut.neuron_effect_blocks import neuron_effect_block


class TestNeuronEffectBlock(unittest.TestCase):

    def setUp(self):
        self.model = UNet()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_neuron_effect_block(self):
        neuron_effect_block(self.model, proportion=0.1)

        # Check that the model's state dict has been modified
        original_state_dict = {k: v.clone().detach() for k, v in self.model.state_dict().items()}
        neuron_effect_block(self.model, proportion=0.1)

        for name, param in self.model.named_parameters():
            if 'weight' in name:
                modified_param = param.data.clone().detach()
                original_param = original_state_dict[name]

                # Check that some weights have been set to zero
                if 'Linear' in name:
                    self.assertTrue((modified_param == 0).any())
                elif 'Conv2d' in name:
                    self.assertTrue((modified_param[:, :int(0.1 * modified_param.shape[1]), :, :] == 0).all())


if __name__ == '__main__':
    unittest.main()
