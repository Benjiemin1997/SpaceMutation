import torch
import unittest

from models.Shufflenetv2.shufflenetv2 import ShuffleNetV2
from mut.random_shuffle import random_shuffle_weight


class TestRandomShuffleWeight(unittest.TestCase):

    def setUp(self):
        self.model = ShuffleNetV2().eval()
        # Initialize or load model weights here

    def tearDown(self):
        del self.model

    def test_random_shuffle_weight(self):
        # Test case to check if the model's weights are shuffled correctly
        original_model_state_dict = self.model.state_dict()

        # Apply the random_shuffle_weight function
        self.model = random_shuffle_weight(self.model)

        # Check if the original state dict is not equal to the new one after shuffling
        self.assertNotEqual(original_model_state_dict, self.model.state_dict())

        # Check if the device of the model remains the same after shuffling
        self.assertEqual(next(iter(original_model_state_dict.values())).device,
                         next(iter(self.model.state_dict().values())).device)

if __name__ == '__main__':
    unittest.main()
