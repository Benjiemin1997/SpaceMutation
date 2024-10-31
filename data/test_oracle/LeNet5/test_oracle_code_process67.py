import unittest

import torch

from models.LeNet5.model_lenet5 import LeNet5
from mut.neuron_effect_blocks import neuron_effect_block


class TestNeuronEffectBlock(unittest.TestCase):

    def setUp(self):
        self.model = LeNet5()
        self.model.eval()

    def test_neuron_effect_block(self):
        # Test case: Check that the model's weights have been modified
        original_model_weights = self.model.state_dict()
        
        # Mutate the model using neuron_effect_block
        mutated_model = neuron_effect_block(self.model)
        
        # Assert that the model has been mutated
        mutated_model_weights = mutated_model.state_dict()
        for key in original_model_weights:
            if torch.equal(original_model_weights[key], mutated_model_weights[key]):
                self.fail(f"Weights for key {key} did not change after mutation.")
        
        # Optional: Check the structure of the mutated model (this is not directly testable with assertions)
        mutated_model_structure = str(mutated_model)
        expected_structure = """LeNet5(
  (0): Sequential(
    (0): Sequential(
      (0): Sequential(
        (0): Conv2d(1, 6, kernel_size=(5, 5), stride=(1, 1))
        (1): Identity()
      )
      (1): Identity()
    )
    (1): GELU(approximate='none')
  )
  (1): Identity()
  (2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (3): Conv2d(6, 16, kernel_size=(5, 5), stride=(1, 1))
  (4): Identity()
  (5): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (6): Flatten(start_dim=1, end_dim=-1)
  (7): Linear(in_features=256, out_features=120, bias=True)
  (8): Identity()
  (9): Sequential(
    (0): Linear(in_features=120, out_features=84, bias=True)
    (1): Identity()
  )
  (10): PReLU(num_parameters=1)
  (11): Linear(in_features=84, out_features=10, bias=True)
  (12): Identity()
)"""
        self.assertEqual(mutated_model_structure, expected_structure)


if __name__ == '__main__':
    unittest.main()