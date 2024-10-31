import torch
import torch.nn as nn

from models.Shufflenetv2.shufflenetv2 import ShuffleNetV2
from mut.neuron_effect_blocks import neuron_effect_block


def test_neuron_effect_block():
    # Initialize a model
    model = ShuffleNetV2()

    # Define expected properties before mutation
    original_layers = list(model.modules())

    # Perform neuron effect block mutation
    mutated_model = neuron_effect_block(model, proportion=0.5)

    # Define expected properties after mutation
    mutated_layers = list(mutated_model.modules())

    # Check that at least one Linear or Conv2d layer has been affected
    assert any(isinstance(layer, nn.Linear) and len(layer.weight.shape) == 2 for layer in mutated_layers), \
        "No Linear layer found with modified weights."

    assert any(isinstance(layer, nn.Conv2d) and len(layer.weight.shape) == 4 for layer in mutated_layers), \
        "No Conv2d layer found with modified weights."

    # Check that the number of affected neurons is approximately the specified proportion
    affected_linear_weights = [layer.weight for layer in mutated_layers if isinstance(layer, nn.Linear)]
    affected_conv_weights = [layer.weight for layer in mutated_layers if isinstance(layer, nn.Conv2d)]
    
    for weights in affected_linear_weights + affected_conv_weights:
        zeros = torch.sum(torch.eq(weights, 0)).item()
        expected_zeros = int(weights.shape[0] * 0.5)
        assert abs(zeros - expected_zeros) <= 5, \
            f"Number of zeros in {weights.shape} does not match expected proportion ({expected_zeros})."

    # Check that the overall structure of the model remains intact
    for i, module in enumerate(original_layers):
        if i < len(mutated_layers):
            assert type(module) == type(mutated_layers[i]), \
                f"Module type mismatch at index {i}: {type(module)} != {type(mutated_layers[i])}."
        else:
            assert len(original_layers) == len(mutated_layers), \
                "Model structure has changed unexpectedly."

    print("All tests passed successfully.")
