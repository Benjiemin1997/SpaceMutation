import unittest

from torch import nn

from models.VGG16.model_vgg16 import VGG16
from mut.random_add_activation import add_activation


class TestAddActivation(unittest.TestCase):
    
    def setUp(self):
        self.model = VGG16()
        self.model = add_activation(self.model)
        
    def test_model_structure(self):
        activation_types = [type(nn.ReLU()), type(nn.LeakyReLU()), type(nn.Sigmoid()), type(nn.Tanh()),
                            type(nn.ELU()), type(nn.PReLU()), type(nn.SELU()), type(nn.GELU())]
        for name, module in self.model.named_modules():
            for child_name, child in module.named_children():
                if isinstance(child, nn.Sequential) and len(child) == 2:
                    _, last_child = child
                    if type(last_child) in activation_types:
                        self.assertTrue(True)
                        return
        self.assertTrue(False, "No activation was added to the model")

    def test_randomness_of_activation(self):
        activation_types = ['ReLU', 'LeakyReLU', 'Sigmoid', 'Tanh', 'ELU', 'PReLU', 'SELU', 'GELU']
        for name, module in self.model.named_modules():
            for child_name, child in module.named_children():
                if isinstance(child, nn.Sequential) and len(child) == 2:
                    _, last_child = child
                    activation_type = type(last_child).__name__
                    self.assertIn(activation_type, activation_types, f"Activation {activation_type} is not one of the expected types: {activation_types}")

if __name__ == '__main__':
    unittest.main()
