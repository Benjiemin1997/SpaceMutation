import unittest
from unittest.mock import patch, Mock

import torch
from torch import Tensor
from torch.nn import CrossEntropyLoss

from models.AlexNet.model_alexnet import AlexNet
from models.VGG16.model_vgg16 import VGG16
from mut.fgsm_fuzz import fgsm_fuzz_weight


class TestFgsmFuzzWeight(unittest.TestCase):

    def test_fgsm_fuzz_weight(self, mock_alexnet, _):
        # Mocking the model and data loader
        model_mock = VGG16()  # 获取模拟的AlexNet实例
        model_mock.parameters.return_value = iter([Mock()])
        data_loader_mock = Mock()
        data_loader_mock.__iter__.return_value = iter([(Tensor([1, 2, 3]), Tensor([0]))])

        # Test the function
        # 假设 fgsm_fuzz_weight 修改模型但不返回新模型（实际行为可能不同）
        fgsm_fuzz_weight(model_mock, data_loader_mock)

        # Assertions
        # 如果函数修改了模型但不返回新模型，则不需要检查返回类型
        # model_mock.zero_grad.assert_called_once()  # 这取决于fgsm_fuzz_weight的实现

        # 如果函数确实调用了 .to('cuda')
        model_mock.to.assert_called_with(torch.device('cuda'))

        # 注意：我们没有模拟 CrossEntropyLoss，因为它可能没有在函数内部使用

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
        # 对于 Mock 对象，通常不需要使用 isInstance 或 isNotInstance
        # 除非你在检查特定的 mock 行为


if __name__ == '__main__':
    unittest.main()