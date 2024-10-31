import torch
import unittest

from models.Shufflenetv2.shufflenetv2 import ShuffleNetV2
from mut.random_shuffle import random_shuffle_weight


class TestRandomShuffleWeight(unittest.TestCase):

    def setUp(self):
        self.model = ShuffleNetV2().to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        # Load your model weights here, for example from a checkpoint or initialization

    def test_random_shuffle_weight(self):
        original_model_state = {param.name: param.data.clone() for param in self.model.parameters()}
        
        # Apply random shuffle to weights
        random_shuffle_weight(self.model)

        # Check that the parameters have been modified
        for name, original_param in original_model_state.items():
            shuffled_param = self.model.state_dict()[name]
            self.assertFalse(torch.allclose(original_param, shuffled_param), 
                             msg=f"Parameter {name} has not been shuffled.")

        # Check that the model still runs without errors after shuffling
        input_data = torch.randn(1, 3, 224, 224).to(self.model.device)
        output_before = self.model(input_data)
        self.model.eval()
        output_after = self.model(input_data)
        self.assertTrue(torch.allclose(output_before, output_after), 
                        msg="The shuffled model produces different outputs than before.")

if __name__ == '__main__':
    unittest.main()
