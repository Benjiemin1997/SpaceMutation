import unittest
from unittest.mock import patch, Mock

import torch
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
        # Create a mock model with specific activation functions
        class MockModel(nn.Module):
            def __init__(self):
                super(MockModel, self).__init__()
                self.layer1 = nn.ReLU()
                self.layer2 = nn.LeakyReLU()
                self.layer3 = nn.Sigmoid()
                self.layer4 = nn.Tanh()
                self.layer5 = nn.ELU()
                self.layer6 = nn.PReLU()
                self.layer7 = nn.SELU()
                self.layer8 = nn.GELU()

        model = MockModel()
        
        # Apply the replace_activations function
        new_model = replace_activations(model)

        # Check that all layers were replaced by random activations
        for name, module in new_model.named_modules():
            if 'layer' in name:
                assert not isinstance(module, (nn.ReLU, nn.LeakyReLU, nn.Sigmoid, nn.Tanh, nn.ELU, nn.PReLU, nn.SELU, nn.GELU)), f"Layer {name} was not replaced."
                
        # Check that the model still runs without errors
        input_data = torch.randn(1, 3, 224, 224)
        output_data = new_model(input_data)
        assert output_data is not None, "The model returned None after activation replacement."

if __name__ == '__main__':
    unittest.main()