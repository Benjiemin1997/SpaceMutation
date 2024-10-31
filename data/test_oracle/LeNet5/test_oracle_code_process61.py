import unittest

from models.LeNet5.model_lenet5 import LeNet5
from mut.guassian_fuzz_splayers import gaussian_fuzzing_splayer


class TestFuzzingMethods(unittest.TestCase):

    def test_fgsm_fuzz_weight(self):
        # Initialize your model and data loader here
        model = LeNet5()

        model_fuzzed = gaussian_fuzzing_splayer(self.model)
        
        # Define expected behavior or properties that should hold after fuzzing
        self.assertTrue(hasattr(model_fuzzed, 'weight'))  # Ensure the model still has its weights
        self.assertNotEqual(model_fuzzed.weight, model.weight)  # Weights should have changed
        
        # Additional checks can be added based on specific requirements

    def test_fgsm_fuzz_weight_mnist(self):
        # Initialize your model and data loader here
        model = LeNet5()

        model_fuzzed = gaussian_fuzzing_splayer(self.model)
        
        # Define expected behavior or properties that should hold after fuzzing
        self.assertTrue(hasattr(model_fuzzed, 'weight'))  # Ensure the model still has its weights
        self.assertNotEqual(model_fuzzed.weight, model.weight)  # Weights should have changed
        
        # Additional checks can be added based on specific requirements

if __name__ == '__main__':
    unittest.main()
