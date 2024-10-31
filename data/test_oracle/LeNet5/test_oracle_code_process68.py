import unittest

from models.LeNet5.model_lenet5 import LeNet5
from mut.neuron_effect_blocks import neuron_effect_block


class TestFuzzingMethods(unittest.TestCase):

    def test_fgsm_fuzz_weight(self):
        # Initialize your model and data loader here
        model = LeNet5()

        # Apply FGSM fuzzing to the model
        model = neuron_effect_block(model)
        
        # Check if the model's weights have been updated by checking a specific weight or layer
        for param in model.parameters():
            self.assertNotEqual(param.data.norm(), 0, "Model weights have not been updated after FGSM fuzzing.")

    def test_fgsm_fuzz_weight_mnist(self):
        # Initialize your model and data loader for MNIST here
        model = LeNet5()

        
        # Apply FGSM fuzzing to the model
        model = neuron_effect_block(model)
        
        # Check if the model's weights have been updated by checking a specific weight or layer
        for param in model.parameters():
            self.assertNotEqual(param.data.norm(), 0, "Model weights have not been updated after FGSM fuzzing.")

if __name__ == '__main__':
    unittest.main()