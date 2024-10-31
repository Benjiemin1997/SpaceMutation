import torch

from mut.random_shuffle import random_shuffle_weight


def test_random_shuffle_weight():

    class MockModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.features = torch.nn.Sequential(
                torch.nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                torch.nn.Identity(),
                torch.nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                torch.nn.Identity(),
                torch.nn.MaxPool2d(kernel_size=2, stride=2),
                torch.nn.Sequential(
                    torch.nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                    torch.nn.ELU(alpha=1.0),
                ),
                torch.nn.Identity(),

                torch.nn.Linear(25088, 100),
            )

        def parameters(self):
            return self.features.parameters()

    model = MockModel()


    shuffled_model = random_shuffle_weight(model)

    assert not torch.allclose(model.features[0].weight, shuffled_model.features[0].weight), "Conv2d layer 1 weights should have been shuffled"
    assert not torch.allclose(model.features[5].weight, shuffled_model.features[5].weight), "Conv2d layer 6 weights should have been shuffled"

    print("Test passed: Random shuffle of model weights was successful.")
