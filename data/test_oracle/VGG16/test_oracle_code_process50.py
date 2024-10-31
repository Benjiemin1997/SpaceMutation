import torch

from mut.random_shuffle import random_shuffle_weight


def test_random_shuffle_weight():

    class MockModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.features = torch.nn.Sequential(
                torch.nn.Conv2d(3, 64, kernel_size=(3, 3)),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(kernel_size=2),
                torch.nn.Conv2d(64, 128, kernel_size=(3, 3)),
                torch.nn.PReLU(),
                torch.nn.Conv2d(128, 256, kernel_size=(3, 3)),
                torch.nn.ReLU(),
                torch.nn.Conv2d(256, 512, kernel_size=(3, 3)),
                torch.nn.SELU(),
                torch.nn.Conv2d(512, 512, kernel_size=(3, 3)),
                torch.nn.SELU(),
                torch.nn.Conv2d(512, 512, kernel_size=(3, 3)),
                torch.nn.Linear(25088, 4096),
                torch.nn.SELU(),
                torch.nn.Dropout(),
                torch.nn.Linear(4096, 100),
            )
    model = MockModel()
    new_model = random_shuffle_weight(model)
    for original_param, new_param in zip(model.parameters(), new_model.parameters()):
        if not torch.allclose(original_param, new_param):
            print("Parameters were successfully shuffled.")

    assert isinstance(new_model, torch.nn.Module)
    assert isinstance(new_model.features, torch.nn.Sequential)
    assert len(new_model.features) == 14
    assert hasattr(new_model.features[0], 'weight') and hasattr(new_model.features[0], 'bias')
    assert hasattr(new_model.features[2], 'weight') and hasattr(new_model.features[2], 'bias')
    assert hasattr(new_model.features[5], 'weight') and hasattr(new_model.features[5], 'bias')
    assert hasattr(new_model.features[7], 'weight') and hasattr(new_model.features[7], 'bias')
    assert hasattr(new_model.features[10], 'weight') and hasattr(new_model.features[10], 'bias')
    assert hasattr(new_model.features[12], 'weight') and hasattr(new_model.features[12], 'bias')
    assert hasattr(new_model.features[14], 'weight') and hasattr(new_model.features[14], 'bias')
    assert hasattr(new_model.features[17], 'weight') and hasattr(new_model.features[17], 'bias')
    assert hasattr(new_model.features[19], 'weight') and hasattr(new_model.features[19], 'bias')
    assert hasattr(new_model.features[21], 'weight') and hasattr(new_model.features[21], 'bias')
    assert hasattr(new_model.features[23], 'weight') and hasattr(new_model.features[23], 'bias')
    assert hasattr(new_model.features[25], 'weight') and hasattr(new_model.features[25], 'bias')
    assert hasattr(new_model.features[27], 'weight') and hasattr(new_model.features[27], 'bias')
    assert hasattr(new_model.features[30], 'weight') and hasattr(new_model.features[30], 'bias')
    assert hasattr(new_model.classifier[0], 'weight') and hasattr(new_model.classifier[0], 'bias')
    assert hasattr(new_model.classifier[2], 'weight') and hasattr(new_model.classifier[2], 'bias')
    assert hasattr(new_model.classifier[4], 'weight') and hasattr(new_model.classifier[4], 'bias')


    print("All tests passed.")
