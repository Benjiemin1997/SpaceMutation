import torch
from torch import nn

from models.ResNet50.model_resnet50 import ResNet50
from mut.guassian_fuzz_splayers import gaussian_fuzzing_splayer
from mut.random_shuffle import random_shuffle_weight
from mut.remove_activation import remove_activations
from mut.replace_activation import replace_activations
from mut.uniform_fuzz import uniform_fuzz_weight

def test_oracle(model, test_loader, device, criterion, num_tests=10):
    mutated_model_gaussian = gaussian_fuzzing_splayer(model)
    print("Gaussian Mutation Applied")
    for inputs, targets in test_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = mutated_model_gaussian(inputs)
        loss = criterion(outputs, targets)
        assert loss.item() <= 0.1, "Gaussian Mutation Test Failed"

    mutated_model_random_shuffle = random_shuffle_weight(model)
    print("Random Shuffle Weights Mutation Applied")
    for inputs, targets in test_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = mutated_model_random_shuffle(inputs)
        loss = criterion(outputs, targets)
        assert loss.item() <= 0.1, "Random Shuffle Weights Mutation Test Failed"
    
    # Remove Activations
    mutated_model_remove_activations = remove_activations(model)
    print("Remove Activations Mutation Applied")
    for inputs, targets in test_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = mutated_model_remove_activations(inputs)
        loss = criterion(outputs, targets)
        assert loss.item() <= 0.1, "Remove Activations Mutation Test Failed"
    
    # Replace Activations
    mutated_model_replace_activations = replace_activations(model)
    print("Replace Activations Mutation Applied")
    for inputs, targets in test_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = mutated_model_replace_activations(inputs)
        loss = criterion(outputs, targets)
        assert loss.item() <= 0.1, "Replace Activations Mutation Test Failed"
    
    # Uniform Fuzzing
    mutated_model_uniform = uniform_fuzz_weight(model)
    print("Uniform Fuzzing Mutation Applied")
    for inputs, targets in test_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = mutated_model_uniform(inputs)
        loss = criterion(outputs, targets)
        assert loss.item() <= 0.1, "Uniform Fuzzing Mutation Test Failed"

