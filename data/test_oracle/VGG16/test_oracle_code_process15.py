import unittest

from torch import nn

from models.VGG16.model_vgg16 import VGG16
from mut.replace_activation import replace_activations

class TestReplaceActivations(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.model = VGG16()

    def test_replace_activations(self):
        mutated_model = replace_activations(self.model)
        for name, module in mutated_model.named_modules():
            if isinstance(module, (nn.ReLU, nn.LeakyReLU, nn.Sigmoid, nn.Tanh, nn.ELU, nn.PReLU, nn.SELU, nn.GELU)):
                _, parent = self._get_parent_module(mutated_model, name)
                new_activation_type = type(getattr(parent, name))
                self.assertNotEqual(new_activation_type, module.__class__)

    def _get_parent_module(self, model, child_name):
        names = child_name.split('.')
        parent_name = names[:-1]
        child_name = names[-1]
        parent = model
        for part in parent_name:
            parent = getattr(parent, part)
        return parent_name, parent

if __name__ == '__main__':
    unittest.main()
