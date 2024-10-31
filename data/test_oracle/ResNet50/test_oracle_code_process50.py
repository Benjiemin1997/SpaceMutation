from models.ResNet50.model_resnet50 import ResNet50
from mut.reverse_activation import reverse_activations


# Test Oracle Code
def test_reverse_activations():

    model = ResNet50()

    reversed_model = reverse_activations(model)
    

    test_cases = [

        (model.conv1, reversed_model.conv1),
        (model.relu, reversed_model.relu),

        (model.fc[0], reversed_model.fc[0]),
    ]
    
    # Run tests
    for original_layer, reversed_layer in test_cases:
        assert original_layer is not reversed_layer, "Layer was not reversed as expected."
        assert isinstance(reversed_layer, type(original_layer)), f"Layer {reversed_layer} is not of the same type as {original_layer}."
        
    print("All tests passed successfully.")
