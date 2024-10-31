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
        mutated_state_dict = mutated_model.state_dict()
        for key in original_state_dict.keys():
            self.assertFalse(torch.equal(original_state_dict[key], mutated_state_dict[key]),
                             msg=f"Weight for {key} did not change after mutation.")
        self.model.load_state_dict(original_state_dict)

if __name__ == '__main__':
    unittest.main()
