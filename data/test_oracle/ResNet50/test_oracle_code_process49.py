import torch
from torch import nn

from models.ResNet50.model_resnet50 import ResNet50
from mut.reverse_activation import reverse_activations


def test_reverse_activations():


    model = ResNet50()
    reversed_model = reverse_activations(model)

    test_cases = [
        (lambda m: isinstance(m, nn.ReLU) and not m.inplace, reversed_model),

        (lambda m: 'register_forward_hook' in dir(m[1]), (reversed_model, reversed_model.children())),

    ]
    

    for test_case, args in test_cases:
        result = test_case(*args)
        assert result, f"Test failed for condition: {test_case.__doc__}"
    
    print("All tests passed successfully!")
