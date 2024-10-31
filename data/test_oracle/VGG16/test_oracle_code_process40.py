import torch
from torch import nn

from models.VGG16.model_vgg16 import VGG16
from mut.random_shuffle import random_shuffle_weight
from mut.remove_activation import remove_activations
from mut.replace_activation import replace_activations
from mut.uniform_fuzz import uniform_fuzz_weight

def test_fgsm_fuzz_weight():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VGG16().to(device)



    fuzzed_model = random_shuffle_weight(model)

    assert isinstance(fuzzed_model, nn.Module), "Output should be an instance of nn.Module"
    

    gaussian_fuzzed_model = random_shuffle_weight(model)
    assert isinstance(gaussian_fuzzed_model, nn.Module), "Gaussian fuzzed model should be an instance of nn.Module"


    shuffled_model = random_shuffle_weight(model)
    assert isinstance(shuffled_model, nn.Module), "Shuffled model should be an instance of nn.Module"


    no_activations_model = remove_activations(model)
    assert isinstance(no_activations_model, nn.Module), "Model without activations should be an instance of nn.Module"


    replaced_activations_model = replace_activations(model)
    assert isinstance(replaced_activations_model, nn.Module), "Model with replaced activations should be an instance of nn.Module"

    # Test for uniform fuzzing
    uniform_fuzzed_model = uniform_fuzz_weight(model)
    assert isinstance(uniform_fuzzed_model, nn.Module), "Uniform fuzzed model should be an instance of nn.Module"

    print("All tests passed successfully.")
