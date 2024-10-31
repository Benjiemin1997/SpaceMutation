import unittest

import torch

from models.Shufflenetv2.shufflenetv2 import ShuffleNetV2
from mut.replace_activation import replace_activations


class TestModelMutations(unittest.TestCase):

    def setUp(self):
        self.model = ShuffleNetV2()  # Initialize your model here

    def test_replace_activations(self):
        original_model = self.model.state_dict()  # Save a copy of the original model's weights

        # Apply mutation
        self.model = replace_activations(self.model)

        # Check that the model has been mutated
        mutated_model = self.model.state_dict()

        # Assertions to verify that the model has indeed been mutated
        for key in original_model:
            original_weight = original_model[key]
            mutated_weight = mutated_model[key]

            # Check that at least one activation function has changed
            if any(isinstance(a, torch.nn.Module) for a in mutated_weight.tolist()):
                self.assertNotEqual(original_weight.tolist(), mutated_weight.tolist())
                break
        else:
            self.fail("No activation function was replaced.")

        # Further assertions can be added based on specific expectations after mutation

if __name__ == '__main__':
    unittest.main()
