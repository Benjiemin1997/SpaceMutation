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

        expected_activation_count = sum([1 for _, m in self.model.named_modules() if isinstance(m, nn.Module)])

        replace_activations(self.model)

        # Assert
        actual_activation_count = sum([1 for _, m in self.model.named_modules() if isinstance(m, nn.Module)])
        self.assertEqual(expected_activation_count, actual_activation_count,
                         "The number of activation functions has changed unexpectedly.")
        
        # Check that activation types have been replaced randomly
        activation_types_before = set([type(m) for _, m in self.model.named_modules() if isinstance(m, nn.Module)])
        activation_types_after = set([type(m) for _, m in self.model.named_modules() if isinstance(m, nn.Module)])
        self.assertNotEqual(activation_types_before, activation_types_after,
                            "All activation types have not been replaced randomly.")

if __name__ == '__main__':
    unittest.main()
