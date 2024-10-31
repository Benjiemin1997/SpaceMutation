import torch
from torch import nn

from models.VGG16.model_vgg16 import VGG16
from mut.guassian_fuzz_splayers import gaussian_fuzzing_splayer
from mut.random_shuffle import random_shuffle_weight
from mut.remove_activation import remove_activations
from mut.replace_activation import replace_activations
from mut.uniform_fuzz import uniform_fuzz_weight

def test_gaussian_fuzzing_splayer():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VGG16().to(device)

    model_fuzzed = gaussian_fuzzing_splayer(model, std_ratio=0.5, target_layer_type=nn.Linear)

    assert len(list(model_fuzzed.named_modules())) != len(list(model.named_modules())), "Model was not modified"

    for name, layer in model_fuzzed.named_modules():
        if isinstance(layer, nn.Linear):
            for param in layer.parameters():
                assert not torch.allclose(param, model.state_dict()[name][param.name]), f"Layer {name} parameters were not modified"
    
    print("Gaussian Fuzzing Splayer test passed.")

def test_random_shuffle_weight():
    linear_layer = nn.Linear(10, 10)
    shuffled_linear_layer = random_shuffle_weight(linear_layer)
    assert not torch.equal(shuffled_linear_layer.weight, linear_layer.weight), "Weight matrix was not shuffled"

    print("Random Shuffle Weight test passed.")

def test_remove_activations():
    model = nn.Sequential(nn.Linear(10, 10), nn.ReLU(), nn.Linear(10, 10))
    model_no_activations = remove_activations(model)
    assert all([not isinstance(m, nn.ModuleList) or all([not isinstance(l, nn.ReLU) for l in m]) for m in model_no_activations.children()]), "Activations were not removed"
    print("Remove Activations test passed.")

def test_replace_activations():
    model = nn.Sequential(nn.Linear(10, 10), nn.ReLU(), nn.Linear(10, 10))
    model_tanh = replace_activations(model)

    assert all([not isinstance(m, nn.ReLU) and isinstance(m, nn.Tanh) for m in model_tanh.modules()]), "Activations were not replaced"

    print("Replace Activations test passed.")

def test_uniform_fuzz_weight():
    linear_layer = nn.Linear(10, 10)
    fuzzed_linear_layer = uniform_fuzz_weight(linear_layer)

    assert not torch.equal(fuzzed_linear_layer.weight, linear_layer.weight), "Weight matrix was not fuzzed"

    print("Uniform Fuzz Weight test passed.")

if __name__ == "__main__":
    test_gaussian_fuzzing_splayer()
    test_random_shuffle_weight()
    test_remove_activations()
    test_replace_activations()
    test_uniform_fuzz_weight()