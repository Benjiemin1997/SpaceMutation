import unittest
from unittest.mock import patch, Mock
from torch import nn

from mut.replace_activation import replace_activations


class TestReplaceActivations(unittest.TestCase):

    @patch('torch.nn.modules.activation.ReLU', new=Mock)
    @patch('torch.nn.modules.activation.LeakyReLU', new=Mock)
    @patch('torch.nn.modules.activation.Sigmoid', new=Mock)
    @patch('torch.nn.modules.activation.Tanh', new=Mock)
    @patch('torch.nn.modules.activation.ELU', new=Mock)
    @patch('torch.nn.modules.activation.PReLU', new=Mock)
    @patch('torch.nn.modules.activation.SELU', new=Mock)
    @patch('torch.nn.modules.activation.GELU', new=Mock)
    
    def test_replace_activations(self):

        class MockModel(nn.Module):
            def __init__(self):
                super(MockModel, self).__init__()
                self.conv = nn.Conv2d(3, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2))
                self.relu = nn.ReLU()
                self.leaky_relu = nn.LeakyReLU()
                self.sigmoid = nn.Sigmoid()
                self.tanh = nn.Tanh()
        
        model = MockModel()

        new_model = replace_activations(model)

        self.assertNotEqual(id(new_model.conv), id(model.conv))
        self.assertNotEqual(id(new_model.relu), id(model.relu))
        self.assertNotEqual(id(new_model.leaky_relu), id(model.leaky_relu))
        self.assertNotEqual(id(new_model.sigmoid), id(model.sigmoid))
        self.assertNotEqual(id(new_model.tanh), id(model.tanh))
        

        self.assertTrue(isinstance(new_model.conv, nn.Conv2d))
        self.assertTrue(isinstance(new_model.relu, nn.ReLU))
        self.assertTrue(isinstance(new_model.leaky_relu, nn.LeakyReLU))
        self.assertTrue(isinstance(new_model.sigmoid, nn.Sigmoid))
        self.assertTrue(isinstance(new_model.tanh, nn.Tanh))

        self.assertNotEqual(type(new_model.conv), type(model.conv))
        self.assertNotEqual(type(new_model.relu), type(model.relu))
        self.assertNotEqual(type(new_model.leaky_relu), type(model.leaky_relu))
        self.assertNotEqual(type(new_model.sigmoid), type(model.sigmoid))
        self.assertNotEqual(type(new_model.tanh), type(model.tanh))

if __name__ == '__main__':
    unittest.main()
