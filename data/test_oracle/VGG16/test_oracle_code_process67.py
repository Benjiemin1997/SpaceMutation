import unittest
from unittest.mock import patch, Mock

import torch
from torch import Tensor

from models.VGG16.model_vgg16 import VGG16
from mut.fgsm_fuzz import fgsm_fuzz_weight


class TestFgsmFuzzWeight(unittest.TestCase):

    def test_fgsm_fuzz_weight(self, mock_alexnet, _):

        model_mock = VGG16()
        model_mock.parameters.return_value = iter([Mock()])
        data_loader_mock = Mock()
        data_loader_mock.__iter__.return_value = iter([(Tensor([1, 2, 3]), Tensor([0]))])

        fgsm_fuzz_weight(model_mock, data_loader_mock)

        model_mock.to.assert_called_with(torch.device('cuda'))


    def test_assertions(self):
        # Test assertions on various conditions
        self.assertIsNone(None)
        self.assertIsNotNone(1)
        self.assertEqual(1, 1)
        self.assertNotEqual(1, 2)
        self.assertTrue(True)
        self.assertFalse(False)
        self.assertIs(True, True)
        self.assertIsNot(False, True)
        self.assertIsInstance(1, int)
        self.assertNotIsInstance('str', int)



if __name__ == '__main__':
    unittest.main()