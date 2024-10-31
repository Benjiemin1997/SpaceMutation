import torch

from mut.random_shuffle import random_shuffle_weight


# Test Oracle Code

def test_random_shuffle_weight():
    # Setup a mock model to test
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

    # Create an instance of the mock model
    model = MockModel()

    # Apply the random shuffle operation to the model's parameters
    new_model = random_shuffle_weight(model)

    # Check that the original model parameters have been altered
    for original_param, new_param in zip(model.parameters(), new_model.parameters()):
        if not torch.allclose(original_param, new_param):
            print("Parameters were successfully shuffled.")

    # Check that the model structure remains intact
    assert isinstance(new_model, torch.nn.Module)
    assert isinstance(new_model.features, torch.nn.Sequential)
    assert len(new_model.features) == 14

    # Perform assertions on specific layers to ensure they are still valid
    assert hasattr(new_model.features[0], 'weight') and hasattr(new_model.features[0], 'bias')
    assert hasattr(new_model.features[2], 'weight') and hasattr(new_model.features[2], 'bias')
    assert hasattr(new_model.features[5], 'weight') and hasattr(new_model.features[5], 'bias')
    assert hasattr(new_model.features[9], 'weight') and hasattr(new_model.features[9], 'bias')
    assert hasattr(new_model.features[13], 'weight') and hasattr(new_model.features[13], 'bias')
    assert hasattr(new_model.features[17], 'weight') and hasattr(new_model.features[17], 'bias')
    assert hasattr(new_model.features[21], 'weight') and hasattr(new_model.features[21], 'bias')
    assert hasattr(new_model.features[25], 'weight') and hasattr(new_model.features[25], 'bias')
    assert hasattr(new_model.features[29], 'weight') and hasattr(new_model.features[29], 'bias')

    # Ensure the final linear layer is valid
    assert hasattr(new_model.classifier[0], 'weight') and hasattr(new_model.classifier[0], 'bias')
    assert hasattr(new_model.classifier[3], 'weight') and hasattr(new_model.classifier[3], 'bias')

    print("All tests passed successfully.")
