import unittest

from torch import nn

from models.Shufflenetv2.shufflenetv2 import ShuffleNetV2
from mut.remove_activation import remove_activations


class TestModelFunction(unittest.TestCase):
    def test_remove_activations(self):
        # Create a model instance or use an existing one for testing purposes
        model = ShuffleNetV2()  # Replace ShuffleNetV2 with your actual model class
        
        # Apply the remove_activations function
        modified_model = remove_activations(model)
        
        # Check if any activation functions remain in the model
        for _, module in modified_model.named_modules():
            self.assertNotIsInstance(module, (nn.ReLU, nn.LeakyReLU, nn.Sigmoid, nn.Tanh)),
            # Add more checks as needed for other activation functions
            
        # Additional tests can be added here to verify specific aspects of the model after modifications
        # For example, you could check the number of parameters, compare outputs before/after, etc.
        
if __name__ == '__main__':
    unittest.main()
