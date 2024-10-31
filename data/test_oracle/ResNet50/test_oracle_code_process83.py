import unittest
from unittest.mock import patch

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
        mock_apply.assert_called()
        self.assertNotEqual(self.model, new_model)

if __name__ == '__main__':
    unittest.main()

