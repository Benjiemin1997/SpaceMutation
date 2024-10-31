import unittest

import torch

from models.VGG16.model_vgg16 import VGG16
from mut.random_add_activation import add_activation


class TestAddActivation(unittest.TestCase):
    
    def setUp(self):
        self.vgg_model = VGG16()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_add_activation(self):
        mutated_model = add_activation(self.vgg_model)
        self.assertNotEqual(mutated_model.features[3][1].__class__.__name__, 'Identity')
        self.assertNotEqual(mutated_model.features[14][1].__class__.__name__, 'Identity')
        self.assertNotEqual(mutated_model.features[21][1].__class__.__name__, 'Identity')
        self.assertNotEqual(mutated_model.features[28][1][1].__class__.__name__, 'Identity')
        self.assertIn(mutated_model.features[3][1].__class__.__name__, ['ReLU', 'LeakyReLU', 'Sigmoid', 'Tanh', 'ELU', 'PReLU', 'SELU', 'GELU'])
        self.assertIn(mutated_model.features[14][1].__class__.__name__, ['ReLU', 'LeakyReLU', 'Sigmoid', 'Tanh', 'ELU', 'PReLU', 'SELU', 'GELU'])
        self.assertIn(mutated_model.features[21][1].__class__.__name__, ['ReLU', 'LeakyReLU', 'Sigmoid', 'Tanh', 'ELU', 'PReLU', 'SELU', 'GELU'])
        self.assertIn(mutated_model.features[28][1][1].__class__.__name__, ['ReLU', 'LeakyReLU', 'Sigmoid', 'Tanh', 'ELU', 'PReLU', 'SELU', 'GELU'])

if __name__ == '__main__':
    unittest.main()
