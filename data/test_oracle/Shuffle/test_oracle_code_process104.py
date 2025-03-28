import unittest
from unittest.mock import patch, Mock

from models.Shufflenetv2.shufflenetv2 import ShuffleNetV2


class TestFgsmFuzzWeight(unittest.TestCase):

    @patch('torch.cuda.is_available', return_value=True)
    @patch('models.AlexNet.model_alexnet.AlexNet')
    def test_fgsm_fuzz_weight(self, mock_alexnet, _):
        # Mocking the model and data loader
        model_mock = ShuffleNetV2() # 获取模拟的AlexNet实例




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