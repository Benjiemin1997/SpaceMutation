import unittest
import torch
import torch.nn as nn

from models.UNet.model_unet import UNet
from mut.neuron_effect_blocks import neuron_effect_block


class TestNeuronEffectBlock(unittest.TestCase):

    def setUp(self):
        self.model = UNet()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.proportion = 0.1

    def test_neuron_effect_block(self):

        input_data = torch.randn(1, 1, 64, 64).to(self.device)


        mutated_model = neuron_effect_block(self.model, self.proportion)

        output_before = self.model(input_data)
        output_after = mutated_model(input_data)

        self.assertFalse(torch.allclose(output_before, output_after),
                         msg="The model output remains unchanged after applying neuron effect block mutation.")

        # Check that some weights have been set to zero
        for layer in mutated_model.modules():
            if isinstance(layer, (nn.Linear, nn.Conv2d)):
                weights = layer.weight.detach().clone().cpu()
                neuron_indices = torch.where(weights == 0)[1]
                self.assertGreaterEqual(neuron_indices.numel(), int(self.proportion * layer.out_features),
                                        msg=f"Not enough neurons were affected by the mutation in layer {layer}")

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
