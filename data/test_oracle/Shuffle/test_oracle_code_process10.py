import torch
from torch import nn

from mut.fgsm_fuzz import fgsm_fuzz_weight
from mut.guassian_fuzz_splayers import gaussian_fuzzing_splayer
from mut.random_shuffle import random_shuffle_weight
from mut.remove_activation import remove_activations
from mut.replace_activation import replace_activations
from mut.uniform_fuzz import uniform_fuzz_weight

def test_oracle():
    # Test Case: FGSM Fuzzing Weight
    def check_fgsm_fuzzing(model, data_loader, epsilon=0.1):
        model = fgsm_fuzz_weight(model, data_loader, epsilon)
    # Test Case: Random Shuffling Weights
    def check_random_shuffling(model):
        model = random_shuffle_weight(model)
    # Test Case: Gaussian Fuzzing Players
    def check_gaussian_fuzzing(model, data_loader):
        model = gaussian_fuzzing_splayer(model, data_loader)
        # Implement assertions to check the impact of Gaussian fuzzing on the model's performance.

    # Test Case: Removing Activations
    def check_activation_removal(model):
        model = remove_activations(model)
        # Implement assertions to confirm that activations have been removed from the model.

    # Test Case: Replacing Activations
    def check_activation_replacement(model):
        model = replace_activations(model)
        # Implement assertions to ensure that the model now uses different activation functions.

    # Test Case: Uniform Fuzzing Weight
    def check_uniform_fuzzing(model, data_loader, epsilon=0.1):
        model = uniform_fuzz_weight(model, data_loader, epsilon)
        # Implement assertions to validate the effect of uniform fuzzing on the model's weights.


