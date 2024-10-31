import unittest

import torch

from models.LeNet5.model_lenet5 import LeNet5
from mut.fgsm_fuzz import fgsm_fuzz_weight
from mut.guassian_fuzz_splayers import gaussian_fuzzing_splayer
from mut.random_shuffle import random_shuffle_weight
from mut.remove_activation import remove_activations
from mut.replace_activation import replace_activations
from mut.uniform_fuzz import uniform_fuzz_weight


class TestFuzzingMethods(unittest.TestCase):

    def test_fgsm_fuzz_weight(self):
        # Initialize your model and data loader here
        model = LeNet5()

        

        model = fgsm_fuzz_weight(model)

        self.assertTrue(any(torch.abs(param.grad).sum() > 0 for param in model.parameters()))
        self.assertTrue(any(param.grad != param.grad.sign() for param in model.parameters()))

    def test_fgsm_fuzz_weight_mnist(self):
        # Initialize your model and data loader here
        model = LeNet5()

        model = fgsm_fuzz_weight(model,)
        

        self.assertTrue(any(torch.abs(param.grad).sum() > 0 for param in model.parameters()))
        self.assertTrue(any(param.grad != param.grad.sign() for param in model.parameters()))

    def test_random_shuffle_weight(self):
        # Initialize your model and data loader here
        model = LeNet5()
        
        # Apply random shuffle weight mutation to the model
        model = random_shuffle_weight(model)

        self.assertNotEqual(id(model.conv2.weight.data), id(model.conv2.weight.data))

    def test_gaussian_fuzz_splayer(self):
        # Initialize your model and data loader here
        model = LeNet5()
        
        # Apply Gaussian fuzzing to the model
        model = gaussian_fuzzing_splayer(model)
        

        self.assertTrue(all(param.grad != 0 for param in model.parameters()))

    def test_uniform_fuzz_weight(self):
        # Initialize your model and data loader here
        model = LeNet5()
        
        # Apply uniform fuzzing to the model
        model = uniform_fuzz_weight(model)
        

        self.assertTrue(all(param.grad != 0 for param in model.parameters()))

    def test_remove_activations(self):
        # Initialize your model and data loader here
        model = LeNet5()
        
        # Apply activation removal mutation to the model
        model = remove_activations(model)
        

        self.assertTrue(hasattr(model, 'remove_activations'))

    def test_replace_activations(self):
        # Initialize your model and data loader here
        model = LeNet5()
        
        # Apply activation replacement mutation to the model
        model = replace_activations(model)

        self.assertTrue(hasattr(model, 'replace_activations'))

if __name__ == '__main__':
    unittest.main()
