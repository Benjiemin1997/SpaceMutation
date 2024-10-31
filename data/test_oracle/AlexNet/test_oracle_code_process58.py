import unittest
from unittest.mock import patch, Mock

from models.AlexNet.model_alexnet import AlexNet
from mut.fgsm_fuzz import fgsm_fuzz_weight


class TestFGSMFuzzWeight(unittest.TestCase):

    def setUp(self):
        self.model = AlexNet()
        self.data_loader = Mock()
        self.epsilon = 0.1

    @patch('your_module.torch.device')
    @patch('your_module.torch.cuda.is_available')
    @patch('your_module.torch.nn.Module.state_dict')
    @patch('your_module.torch.autograd.set_detect_anomaly')
    def test_fgsm_fuzz_weight(self, mock_set_detect_anomaly, mock_state_dict, mock_cuda_available, mock_device):
        mock_cuda_available.return_value = True
        mock_device.return_value = 'cuda'
        mock_state_dict.return_value = {'model_state': {}}
        mock_set_detect_anomaly.return_value = None

        # Mock the data_loader to return dummy tensors
        images_mock = Mock()
        labels_mock = Mock()
        self.data_loader.__iter__.return_value = [(images_mock, labels_mock)]

        # Call the function under test
        mutated_model = fgsm_fuzz_weight(self.model, self.data_loader, self.epsilon)

        # Assertions
        self.assertEqual(mutated_model, self.model)  # The model should be mutated in place
        mock_state_dict.assert_called_once()  # state_dict should be called once
        mock_set_detect_anomaly.assert_called_once()  # set_detect_anomaly should be called once

if __name__ == '__main__':
    unittest.main()
