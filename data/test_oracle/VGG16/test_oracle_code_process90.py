import torch
from torch import nn
from torch.nn import PReLU

from models.VGG16.model_vgg16 import VGG16
from mut.random_shuffle import random_shuffle_weight
from mut.remove_activation import remove_activations
from mut.replace_activation import replace_activations
from mut.uniform_fuzz import uniform_fuzz_weight

def test_fgsm_fuzz_weight():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VGG16().to(device)
    assert isinstance(model, nn.Module), "Model should be an instance of nn.Module"
    assert model.vgg.features[10].activation == PReLU(), "PReLU should remain unchanged after processing"

    model = random_shuffle_weight(model)
    assert model.vgg.features[0].weight.shape != model.vgg.features[0].weight_orig.shape, "Weights should have been shuffled"
    
    model = remove_activations(model)
    assert model.vgg.features[0].activation is None, "Activation should have been removed"
    
    model = replace_activations(model)
    assert model.vgg.features[0].activation != PReLU(), "Activation should have been replaced"
    
    model = uniform_fuzz_weight(model)
    assert torch.all(torch.abs(model.vgg.features[0].weight - model.vgg.features[0].weight_orig) <= 0.1), "Weights should be within [-0.1, 0.1]"
    
    print("All tests passed successfully.")
