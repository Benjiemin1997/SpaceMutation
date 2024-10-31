import torch
from torch import nn

from models.ResNet50.model_resnet50 import ResNet50
from mut.guassian_fuzz_splayers import gaussian_fuzzing_splayer
from mut.random_shuffle import random_shuffle_weight
from mut.remove_activation import remove_activations
from mut.replace_activation import replace_activations
from mut.uniform_fuzz import uniform_fuzz_weight

def test_fgsm_fuzz_weight():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet50().to(device)

    model_gaussian_fuzzed = gaussian_fuzzing_splayer(model)
    assert torch.allclose(model_gaussian_fuzzed.conv1.weight, model.conv1.weight + torch.randn_like(model.conv1.weight), atol=1e-3)

    model_random_shuffled = random_shuffle_weight(model)
    assert not torch.equal(model_random_shuffled.conv1.weight, model.conv1.weight)


    model_removed_activations = remove_activations(model)
    assert len(model_removed_activations.modules()) < len(model.modules())


    model_replaced_activations = replace_activations(model)
    assert type(model_replaced_activations.relu) != nn.ReLU


    model_uniform_fuzzed = uniform_fuzz_weight(model)
    assert torch.allclose(model_uniform_fuzzed.conv1.weight, model.conv1.weight + torch.rand_like(model.conv1.weight) * 2, atol=1e-3)

    print("All fuzzing methods tested successfully.")