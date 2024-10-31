import unittest

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

            # Verify that the model's state_dict has been altered
            for layer in ['linear_layers', 'conv_layers']:
                for name, param in modified_model.named_parameters():
                    if layer in name:
                        self.assertNotEqual(original_model[name].tolist(), param.tolist())

            # Optionally, you can verify specific layers or neurons here
            # For example, check if a specific neuron is set to 0
            neuron_to_check = 'layer_name_neuron_index'
            if neuron_to_check in modified_model:
                self.assertEqual(0, modified_model[neuron_to_check])


            mutated_model = gaussian_fuzzing_splayer(modified_model)

            self.model.load_state_dict(original_model)

if __name__ == '__main__':
    unittest.main()
