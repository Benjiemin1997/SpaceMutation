import torch
import unittest

from mut.uniform_fuzz import uniform_fuzz_weight


class TestModelMutation(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg16', pretrained=True).to(self.device)

    def tearDown(self):
        del self.model

    def test_uniform_fuzz_weight(self):
        model = self.model
        model = uniform_fuzz_weight(model)
        for param in model.parameters():
            if param.requires_grad:
                self.assertTrue(torch.allclose(param.data, uniform_fuzz_weight(model, -0.1, 0.1), atol=1e-3))

if __name__ == '__main__':
    unittest.main()
