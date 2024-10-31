import torch
import unittest

from models.Shufflenetv2.shufflenetv2 import ShuffleNetV2


class TestUniformFuzzWeight(unittest.TestCase):

    def setUp(self):
        self.model = ShuffleNetV2().eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def tearDown(self):
        del self.model

    def test_uniform_fuzz_weight(self):
        # Given
        lower_bound = -0.1
        upper_bound = 0.1
        expected_shape = [param.data.shape for param in self.model.parameters()]

        # When
        self.model = uniform_fuzz_weight(self.model, lower_bound, upper_bound)

        # Then
        for param in self.model.parameters():
            self.assertEqual(param.data.shape, expected_shape)
            self.assertTrue(torch.all(param.data >= lower_bound))
            self.assertTrue(torch.all(param.data <= upper_bound))

if __name__ == '__main__':
    unittest.main()