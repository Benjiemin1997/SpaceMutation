import torch
import unittest
from torch import nn

from models.ResNet50.model_resnet50 import ResNet50
from mut.replace_activation import replace_activations


class TestFuzzWeight(unittest.TestCase):

    def setUp(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ResNet50().to(self.device)

    def test_fgsm_fuzz_weight(self):
        # Fuzzing by replacing activations
        model_replaced_activations = replace_activations(self.model)

        found_relu = False
        for name, module in model_replaced_activations.named_modules():
            if 'relu' in name and isinstance(module, nn.ReLU):
                found_relu = True
                break
        self.assertFalse(found_relu, "The 'relu' layer should not be of type nn.ReLU after replacement.")

        self.assertNotIsInstance(model_replaced_activations.layer1[0].relu, nn.ReLU)

    def tearDown(self):
        # 清理资源
        pass


if __name__ == '__main__':
    unittest.main()