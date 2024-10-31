import torch
import unittest

from models.VGG16.model_vgg16 import VGG16
from mut.uniform_fuzz import uniform_fuzz_weight


class TestModelMutation(unittest.TestCase):
    def setUp(self):

        self.model = VGG16()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_uniform_fuzz_weight(self):

        self.model = uniform_fuzz_weight(self.model)

        # Define inputs for the model
        input_data = torch.randn(1, 3, 224, 224).to(self.device)

        # Run the model before and after mutation
        output_before = self.model(input_data).detach().cpu().numpy()
        _ = self.model.to(self.device)
        output_after = self.model(input_data).detach().cpu().numpy()

        self.assertFalse(torch.allclose(output_before, output_after),
                         "Output did not change after applying the uniform fuzz weight mutation.")

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
