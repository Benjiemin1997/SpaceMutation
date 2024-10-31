import torch
import unittest

from mut.guassian_fuzz_splayers import gaussian_fuzzing_splayer
from mut.random_shuffle import random_shuffle_weight
from mut.remove_activation import remove_activations
from mut.replace_activation import replace_activations
from mut.uniform_fuzz import uniform_fuzz_weight


class TestModelMutation(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = torch.hub.load('pytorch/vision:v0.6.0', 'vgg16', pretrained=True).to(self.device)

    def tearDown(self):
        del self.model
        torch.cuda.empty_cache()

    def test_uniform_fuzz_weight(self):

        new_model = uniform_fuzz_weight(self.model)
        for param in new_model.parameters():

            self.assertFalse(torch.allclose(param, self.model.state_dict()[param.name]))

    def test_mutations_integration(self):

        self.model = torch.hub.load('pytorch/vision:v0.6.0', 'vgg16', pretrained=True).to(self.device)

        # Gaussian fuzzing
        self.model = gaussian_fuzzing_splayer(self.model)

        # Random shuffle of weights
        self.model = random_shuffle_weight(self.model)

        # Remove activations
        self.model = remove_activations(self.model)

        self.model = replace_activations(self.model)

        self.model = uniform_fuzz_weight(self.model)


        for param in self.model.parameters():
            self.assertFalse(torch.allclose(param, self.model.state_dict()[param.name]))

if __name__ == '__main__':
    unittest.main()