import torch
from torch import nn
from mut.guassian_fuzz_splayers import gaussian_fuzzing_splayer
from mut.random_shuffle import random_shuffle_weight
from mut.remove_activation import remove_activations
from mut.replace_activation import replace_activations
from mut.uniform_fuzz import uniform_fuzz_weight

def test_oracle(model, data_loader, epsilon=0.1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Test Gaussian Fuzzing
    print("\nGaussian Fuzzing Test:")
    gaussian_fuzzing_splayer(model, data_loader, epsilon)
    # Check if the model has been fuzzed as expected
    
    # Test Random Weight Shuffling
    print("\nRandom Weight Shuffling Test:")
    random_shuffle_weight(model)
    # Check if the weights have been shuffled randomly
    
    # Test Removing Activations
    print("\nRemoving Activations Test:")
    remove_activations(model)
    # Check if activations have been removed as expected
    
    # Test Replacing Activations
    print("\nReplacing Activations Test:")
    replace_activations(model)
    # Check if activations have been replaced as expected
    
    # Test Uniform Fuzzing
    print("\nUniform Fuzzing Test:")
    uniform_fuzz_weight(model, data_loader, epsilon)
