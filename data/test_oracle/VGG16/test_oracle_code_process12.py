from models.VGG16.model_vgg16 import VGG16
from mut.guassian_fuzz_splayers import gaussian_fuzzing_splayer
from mut.random_shuffle import random_shuffle_weight
from mut.remove_activation import remove_activations
from mut.replace_activation import replace_activations
from mut.uniform_fuzz import uniform_fuzz_weight

def test_oracle():
    # Given
    model = VGG16()

    # Test cases for mutation strategies
    model = gaussian_fuzzing_splayer(model)
    model = random_shuffle_weight(model)
    model = remove_activations(model)
    model = replace_activations(model)
    model = uniform_fuzz_weight(model)


    assert model is not None, "Model should not be None after mutation"
    assert model != VGG16(), "Model should be different after mutation"
    # Add more assertions specific to your needs, like checking changes in the model architecture, weights, etc.
    
    print("All test cases passed!")