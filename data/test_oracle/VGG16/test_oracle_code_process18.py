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
    original_params = {id(param): param.data.clone() for param in model.parameters()}
    shuffled_params = {id(param): param.data.clone() for param in shuffled_model.parameters()}

    for original_param_id, original_param in original_params.items():
        shuffled_param = shuffled_params.get(original_param_id)
        if shuffled_param is None:
            raise AssertionError(f"No shuffled parameter found for id: {original_param_id}")
        try:
            torch.testing.assert_allclose(original_param, shuffled_param, atol=1e-4)
        except AssertionError as e:
            print(f"Shuffled parameter mismatch at id: {original_param_id}")
            raise e

    print("Random shuffle of model weights passed all checks.")
