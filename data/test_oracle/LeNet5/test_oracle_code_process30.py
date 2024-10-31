import torch
import unittest

from models.LeNet5.model_lenet5 import LeNet5
from mut.uniform_fuzz import uniform_fuzz_weight


class TestUniformFuzzWeight(unittest.TestCase):

    def setUp(self):
        self.model = LeNet5()
        self.lower_bound = -0.1
        self.upper_bound = 0.1

    def test_model_uniform_fuzz(self):
        # Apply uniform fuzzing to the model's weights
        self.model = uniform_fuzz_weight(self.model, self.lower_bound, self.upper_bound)
        for param in self.model.parameters():
            if param.requires_grad:
                self.assertTrue(torch.std(param.data).item() > 0, 
                                msg="Parameter {} did not change after uniform fuzzing".format(param))

        original_state_dict = {k: v.clone() for k, v in self.model.state_dict().items()}
        self.model = uniform_fuzz_weight(self.model, self.lower_bound, self.upper_bound)
        updated_state_dict = self.model.state_dict()
        self.assertNotEqual(original_state_dict, updated_state_dict,
                            msg="Model state_dict did not update after uniform fuzzing")

if __name__ == '__main__':
    unittest.main()
