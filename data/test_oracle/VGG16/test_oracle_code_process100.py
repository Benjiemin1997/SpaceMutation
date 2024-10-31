import torch
import unittest

from mut.uniform_fuzz import uniform_fuzz_weight


class TestModelMutation(unittest.TestCase):
    def setUp(self):
        from torchvision.models import vgg16
        self.model = vgg16(pretrained=True).eval()

    def test_uniform_fuzz_weight(self):
        original_state_dict = self.model.state_dict()

        mutated_model = uniform_fuzz_weight(self.model)

        for name, param in mutated_model.named_parameters():
            if 'weight' in name:
                self.assertNotEqual(param.data.sum(), original_state_dict[name].sum())

        self.assertEqual(len(mutated_model.features), len(self.model.features))
        self.assertEqual(len(mutated_model.classifier), len(self.model.classifier))

    def tearDown(self):
        del self.model

if __name__ == '__main__':
    unittest.main()
