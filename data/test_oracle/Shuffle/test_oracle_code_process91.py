import unittest

from models.Shufflenetv2.shufflenetv2 import ShuffleNetV2
from mut.remove_activation import remove_activations


class TestModelFunction(unittest.TestCase):
    def test_remove_activations(self):
        # Create a model instance or use an existing one for testing purposes
        model = ShuffleNetV2()  # Replace with actual model instantiation
        
        # Apply the remove_activations function to the model
        modified_model = remove_activations(model)

        # Assert that the number of modules has changed
        original_num_modules = sum(p.numel() for p in model.parameters() if p.requires_grad)
        modified_num_modules = sum(p.numel() for p in modified_model.parameters() if p.requires_grad)
        
        self.assertLess(modified_num_modules, original_num_modules, 
                        "The number of trainable parameters should decrease after removing activations.")
        
        # Assert that some activations have been removed
        expected_activation_count = len([m for m in model.modules() if isinstance(m, (nn.ReLU, nn.LeakyReLU, nn.Sigmoid, nn.Tanh))])
        actual_activation_count = len([m for m in modified_model.modules() if isinstance(m, (nn.ReLU, nn.LeakyReLU, nn.Sigmoid, nn.Tanh))])
        
        self.assertLess(actual_activation_count, expected_activation_count,
                        "Some activations should have been removed from the model.")
        
        # Additional assertions based on specific behaviors or outputs can be added here
        # For example, you might check that certain layers still work correctly after modifications
    
if __name__ == '__main__':
    unittest.main()