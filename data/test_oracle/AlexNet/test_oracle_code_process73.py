import unittest
from unittest.mock import patch, Mock

from models.AlexNet.model_alexnet import AlexNet
from mut.fgsm_fuzz import fgsm_fuzz_weight


class TestFGSMFuzzWeight(unittest.TestCase):

    @patch('torch.cuda.is_available', return_value=True)
    def setUp(self, _):
        self.device = 'cuda'
        self.model = AlexNet()
        self.data_loader = Mock()
        self.epsilon = 0.1

    @patch('your_module.fgsm_fuzz_weight')
    def test_fgsm_fuzz_weight(self, mock_fgsm_fuzz_weight):
        # Arrange
        expected_model = self.model.clone()

        # Act
        actual_model = fgsm_fuzz_weight(self.model, self.data_loader, self.epsilon)

        # Assert
        mock_fgsm_fuzz_weight.assert_called_once_with(self.model, self.data_loader, self.epsilon)
        self.assertEqual(actual_model, expected_model)

if __name__ == '__main__':
    unittest.main()
