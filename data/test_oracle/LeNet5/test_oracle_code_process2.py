import unittest

from torch import nn
from torch.utils.data import DataLoader

from models.LeNet5.model_lenet5 import LeNet5
from mut.random_add_activation import add_activation
from util.dataset import CustomCIFAR100Dataset


class TestAddActivation(unittest.TestCase):
    
    def setUp(self):
        self.model = LeNet5()

        
    def test_add_activation(self):
        modified_model = add_activation(self.model)

        self.assertTrue(any(isinstance(module, nn.Module) for module in modified_model.modules()))
        
    def test_dataloader(self):
        for images, labels in self.dataloader:
            pass
        # Ensure the dataloader successfully loads data from the dataset
        self.assertTrue(images.size(0) == 16)
        self.assertTrue(labels.size(0) == 16)
        
if __name__ == '__main__':
    unittest.main()