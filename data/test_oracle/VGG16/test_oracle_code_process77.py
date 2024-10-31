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
                        break
            else:
                continue
            break
        else:
            self.fail("No activation added to the model")

    def test_model_performance(self):
        pass

if __name__ == '__main__':
    unittest.main()
