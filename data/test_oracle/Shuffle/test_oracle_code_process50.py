import unittest

from models.Shufflenetv2.shufflenetv2 import ShuffleNetV2
from mut.remove_activation import remove_activations


class TestModelFunction(unittest.TestCase):
    def test_remove_activations(self):
        # Create a model instance or use an existing one for testing purposes
        model = ShuffleNetV2()  # Replace ShuffleNetV2 with your actual model class
        
        # Apply the remove_activations function
        modified_model = remove_activations(model)
        
        # Check if the model's structure has been altered as expected
        # You might want to define specific assertions based on your needs
        # For example, check if certain activation functions have been removed
        self.assertNotIn('ReLU', str(modified_model))
        self.assertNotIn('LeakyReLU', str(modified_model))
        self.assertNotIn('Sigmoid', str(modified_model))
        self.assertNotIn('Tanh', str(modified_model))

        # Additional checks can be added here depending on the specifics of the model and the desired outcome

if __name__ == '__main__':
    unittest.main()
