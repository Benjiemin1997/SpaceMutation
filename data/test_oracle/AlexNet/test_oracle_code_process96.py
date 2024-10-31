import torch
import unittest

from models.AlexNet.model_alexnet import AlexNet
from mut.uniform_fuzz import uniform_fuzz_weight


class TestUniformFuzzWeight(unittest.TestCase):

    def setUp(self):
        self.model = AlexNet()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_uniform_fuzz_weight(self):
        # Test case for the uniform fuzz weight function
        original_state_dict = self.model.state_dict()

        # Apply the uniform fuzzing to the model's weights
        uniform_fuzz_weight(self.model)

        # Check that all parameters have been modified
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.assertNotEqual(param.data, original_state_dict[name].data)

        # Check that the model's state dict has changed
        new_state_dict = self.model.state_dict()
        self.assertNotEqual(new_state_dict, original_state_dict)

if __name__ == '__main__':
    unittest.main()
