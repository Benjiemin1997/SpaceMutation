import torch

from models.Shufflenetv2.shufflenetv2 import ShuffleNetV2
from mut.guassian_fuzz_splayers import gaussian_fuzzing_splayer
from mut.random_shuffle import random_shuffle_weight
from mut.remove_activation import remove_activations
from mut.replace_activation import replace_activations
from mut.uniform_fuzz import uniform_fuzz_weight

# Test case for the model mutation functions

def test_random_shuffle_weight():
    # Initialize a model
    model = ShuffleNetV2()

    # Save original model parameters
    original_params = {name: param.clone() for name, param in model.named_parameters()}

    # Apply the mutation function
    mutated_model = random_shuffle_weight(model)

    # Check that the model has been mutated
    assert not torch.allclose(original_params[next(iter(original_params))], mutated_model.state_dict()[next(iter(mutated_model.state_dict()))])

    # Restore original model parameters (optional)
    mutated_model.load_state_dict(original_params)

def test_gaussian_fuzzing_splayer():
    # Initialize a model
    model = ShuffleNetV2()

    # Apply Gaussian fuzzing to a layer's weights
    gaussian_fuzzing_splayer(model, layer_name="pre.0", std_dev=0.1)

    # Check if any of the weights have been altered by comparing the mean and std deviation before and after
    layer_weights = next(p for n, p in model.named_parameters() if "pre.0" in n)
    assert layer_weights.mean().item() != layer_weights.mean().item(), "No change detected in layer weights"

def test_uniform_fuzz_weight():
    # Initialize a model
    model = ShuffleNetV2()

    # Apply uniform fuzzing to a layer's weights
    uniform_fuzz_weight(model, layer_name="pre.0", low=-0.1, high=0.1)

    # Check if any of the weights have been altered by comparing the min and max values before and after
    layer_weights = next(p for n, p in model.named_parameters() if "pre.0" in n)
    assert layer_weights.min().item() != layer_weights.min().item(), "No change detected in layer weights"
    assert layer_weights.max().item() != layer_weights.max().item(), "No change detected in layer weights"

def test_remove_activations():
    # Initialize a model
    model = ShuffleNetV2()

    # Remove activations from the first ShuffleUnit in stage2
    remove_activations(model, layer_name="stage2.1.residual.7")

    # Check if the activation is removed by checking for the existence of the activation function
    residual_layer = next((n, p) for n, p in model.named_parameters() if "stage2.1.residual.7" in n)
    assert "GELU" not in str(type(residual_layer[1])), "Activation was not removed"

def test_replace_activations():
    # Initialize a model
    model = ShuffleNetV2()

    # Replace activations in the first ShuffleUnit in stage2 with a new activation function
    replace_activations(model, layer_name="stage2.1.residual.7", new_activation="LeakyReLU")

    # Check if the activation is replaced by verifying the type of the activation function
    residual_layer = next((n, p) for n, p in model.named_parameters() if "stage2.1.residual.7" in n)
    assert isinstance(residual_layer[1], torch.nn.LeakyReLU), "Activation was not replaced"