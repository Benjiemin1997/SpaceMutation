import unittest
import torch
from torch.utils.data import DataLoader

from models.ResNet50.model_resnet50 import ResNet50
from mut.fgsm_fuzz import fgsm_fuzz_weight, fgsm_fuzz_weight_mnist


class TestFGSMFuzzWeight(unittest.TestCase):
    def setUp(self):
        self.model = ResNet50()
        self.data_loader = DataLoader(torch.randn(10, 1, 28, 28), batch_size=10, shuffle=True)
        self.epsilon = 0.1

    def test_fgsm_fuzz_weight(self):
        mutated_model = fgsm_fuzz_weight(self.model, self.data_loader, self.epsilon)
        self.assertIsNotNone(mutated_model)

    def test_fgsm_fuzz_weight_mnist(self):
        mutated_model = fgsm_fuzz_weight_mnist(self.model, self.data_loader, self.epsilon)
        self.assertIsNotNone(mutated_model)

    def test_data_loader(self):
        for images, labels in self.data_loader:
            self.assertTrue(torch.is_tensor(images))
            self.assertTrue(torch.is_tensor(labels))
            self.assertEqual(images.shape, torch.Size([10, 1, 28, 28]))
            self.assertEqual(labels.shape, torch.Size([10]))

if __name__ == '__main__':
    unittest.main()