import unittest
from unittest.mock import patch

from torch import nn

from models.ResNet50.model_resnet50 import ResNet50
from mut.replace_activation import replace_activations


class TestReplaceActivations(unittest.TestCase):

    def setUp(self):
        self.model = ResNet50()
        self.original_model = self.model.state_dict().copy()

    def tearDown(self):
        self.model.load_state_dict(self.original_model)

    @patch('torch.nn.Module.apply')
    def test_replace_activations(self, mock_apply):
        new_model = replace_activations(self.model)
        

        self.assertNotEqual(self.model, new_model)
        original_activations_count = sum(1 for _, module in self.model.named_modules() if isinstance(module, (nn.ReLU, nn.LeakyReLU, nn.Sigmoid, nn.Tanh, nn.ELU, nn.PReLU, nn.SELU, nn.GELU)))
        new_activations_count = sum(1 for _, module in new_model.named_modules() if isinstance(module, (nn.ReLU, nn.LeakyReLU, nn.Sigmoid, nn.Tanh, nn.ELU, nn.PReLU, nn.SELU, nn.GELU)))
        self.assertNotEqual(original_activations_count, new_activations_count)
        self.assertNotEqual(self.model.state_dict(), new_model.state_dict())

        mock_apply.assert_called()

if __name__ == '__main__':
    unittest.main()
