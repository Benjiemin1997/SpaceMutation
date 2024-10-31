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

    for original_param, shuffled_param in zip(model.parameters(), new_model.parameters()):
        assert not torch.allclose(original_param, shuffled_param), "Model parameters should be shuffled"

    print("Random shuffle weight test passed successfully.")
