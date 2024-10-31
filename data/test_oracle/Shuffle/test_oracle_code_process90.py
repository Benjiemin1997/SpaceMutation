import torch
import torch.nn as nn
import random
from mut.guassian_fuzz_splayers import gaussian_fuzzing_splayer
from mut.random_add_activation import add_activation
from mut.random_shuffle import random_shuffle_weight
from mut.remove_activation import remove_activations
from mut.replace_activation import replace_activations
from mut.uniform_fuzz import uniform_fuzz_weight

def test_model(model, input_data, expected_output=None, device='cuda'):
    """
    Test the given model against an input data sample.

    Args:
        model: The model to test.
        input_data: The input data to pass through the model.
        expected_output: Expected output for comparison. If None, no comparison is made.
        device: Device to run the model on ('cuda' or 'cpu').
    """
    model.to(device)
    input_data = input_data.to(device)
    
    # Forward pass
    output = model(input_data)
    
    # Check if expected output is provided and compare actual output
    if expected_output is not None:
        assert torch.allclose(output, expected_output, atol=1e-2), "Output does not match expected result."
    
    # Clean up
    del input_data
    del output

def test_add_activation(model, input_data, expected_output=None, device='cuda'):
    """
    Test if adding random activation function to the model changes its output.

    Args:
        model: The model to modify and test.
        input_data: The input data to pass through the modified model.
        expected_output: Expected output for comparison. If None, no comparison is made.
        device: Device to run the model on ('cuda' or 'cpu').
    """
    modified_model = add_activation(model)
    test_model(modified_model, input_data, expected_output, device)

def test_random_shuffle(model, input_data, expected_output=None, device='cuda'):
    """
    Test if randomly shuffling the weights of the model changes its output.

    Args:
        model: The model to modify and test.
        input_data: The input data to pass through the modified model.
        expected_output: Expected output for comparison. If None, no comparison is made.
        device: Device to run the model on ('cuda' or 'cpu').
    """
    modified_model = random_shuffle_weight(model)
    test_model(modified_model, input_data, expected_output, device)

def test_remove_activations(model, input_data, expected_output=None, device='cuda'):
    """
    Test if removing all activation functions from the model changes its output.

    Args:
        model: The model to modify and test.
        input_data: The input data to pass through the modified model.
        expected_output: Expected output for comparison. If None, no comparison is made.
        device: Device to run the model on ('cuda' or 'cpu').
    """
    modified_model = remove_activations(model)
    test_model(modified_model, input_data, expected_output, device)

def test_replace_activations(model, input_data, expected_output=None, device='cuda'):
    """
    Test if replacing all activation functions with random ones changes the model's output.

    Args:
        model: The model to modify and test.
        input_data: The input data to pass through the modified model.
        expected_output: Expected output for comparison. If None, no comparison is made.
        device: Device to run the model on ('cuda' or 'cpu').
    """
    modified_model = replace_activations(model)
    test_model(modified_model, input_data, expected_output, device)

def test_uniform_fuzz(model, input_data, expected_output=None, device='cuda', fuzz_factor=0.1):
    """
    Test if uniformly fuzzing the weights of the model changes its output.

    Args:
        model: The model to modify and test.
        input_data: The input data to pass through the modified model.
        expected_output: Expected output for comparison. If None, no comparison is made.
        device: Device to run the model on ('cuda' or 'cpu').
        fuzz_factor: Factor by which to fuzz the weights.
    """
    modified_model = uniform_fuzz_weight(model, fuzz_factor)
    test_model(modified_model, input_data, expected_output, device)
