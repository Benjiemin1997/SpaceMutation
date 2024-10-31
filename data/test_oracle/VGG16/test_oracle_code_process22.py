import torch
from torch import nn

from models.VGG16.model_vgg16 import VGG16
from mut.fgsm_fuzz import fgsm_fuzz_weight
from mut.guassian_fuzz_splayers import gaussian_fuzzing_splayer
from mut.random_shuffle import random_shuffle_weight
from mut.remove_activation import remove_activations
from mut.replace_activation import replace_activations
from mut.uniform_fuzz import uniform_fuzz_weight

def test_fgsm_fuzz_weight_model():
    model = VGG16()

    data_loader = torch.utils.data.DataLoader(
        dataset=torch.randn(10, 3, 224, 224),
        batch_size=1,
        shuffle=True
    )

    model = fgsm_fuzz_weight(model, data_loader)

    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            assert not torch.allclose(module.weight, torch.zeros_like(module.weight)), \
                f"Weight of {name} has not been fuzzed."

    model = gaussian_fuzzing_splayer(model)
    model = random_shuffle_weight(model)
    model = remove_activations(model)
    model = replace_activations(model)
    model = uniform_fuzz_weight(model)

    assert model(torch.randn(1, 3, 224, 224)).shape == (1, 100), "Model output shape after mutation is incorrect."