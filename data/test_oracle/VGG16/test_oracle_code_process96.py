import torch
import unittest

from mut.uniform_fuzz import uniform_fuzz_weight


class TestModelMutation(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = torch.hub.load('pytorch/vision:v0.9.0', 'vgg16', pretrained=True).to(self.device)

    def tearDown(self):
        del self.model
        torch.cuda.empty_cache()
    def test_uniform_fuzz_weight(self):
        new_model = uniform_fuzz_weight(self.model, lower_bound=-0.1, upper_bound=0.1)

        for param in new_model.parameters():
            if param.grad is not None:
                self.assertTrue(torch.any(param.grad != 0))

        original_state_dict = {k: v.clone() for k, v in self.model.state_dict().items()}
        new_state_dict = {k: v.clone() for k, v in new_model.state_dict().items()}
        self.assertNotEqual(original_state_dict, new_state_dict)

if __name__ == '__main__':
    unittest.main()
