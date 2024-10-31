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
                torch.nn.MaxPool2d(kernel_size=2),
                torch.nn.Conv2d(256, 512, kernel_size=(3, 3)),
                torch.nn.SELU(),
                torch.nn.MaxPool2d(kernel_size=2),
                torch.nn.Conv2d(512, 512, kernel_size=(3, 3)),
                torch.nn.SELU(),
                torch.nn.Conv2d(512, 512, kernel_size=(3, 3)),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(kernel_size=2),
                torch.nn.Linear(25088, 4096),
                torch.nn.SELU(),
                torch.nn.Dropout(p=0.5),
                torch.nn.Linear(4096, 100)
            )

        def forward(self, x):
            x = self.features(x)
            return x

    model = MockModel()
    model = random_shuffle_weight(model)
    for param in model.parameters():
        assert not torch.allclose(param.data, torch.zeros_like(param.data)), "Weights should have been shuffled."

    print("Test Passed: Weights successfully shuffled.")

# Run the test
test_random_shuffle_weight()