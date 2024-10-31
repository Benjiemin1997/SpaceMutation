import torch

from mut.random_shuffle import random_shuffle_weight


# Test Oracle Code

def test_random_shuffle_weight():
    # Create a mock model to test
    class MockModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(3, 3))
            self.conv2 = torch.nn.Conv2d(64, 128, kernel_size=(3, 3))
            self.conv3 = torch.nn.Conv2d(128, 256, kernel_size=(3, 3))
            self.conv4 = torch.nn.Conv2d(256, 512, kernel_size=(3, 3))
            self.linear1 = torch.nn.Linear(64, 10)

        def forward(self, x):
            x = torch.relu(self.conv1(x))
            x = torch.relu(self.conv2(x))
            x = torch.relu(self.conv3(x))
            x = torch.relu(self.conv4(x))
            x = self.linear1(x)
            return x

    # Initialize the mock model
    model = MockModel()

    # Apply random shuffle weight
    model = random_shuffle_weight(model)


    print("Random Shuffle Weight Test Passed")

def test_random_shuffle_exception_predictor():
    # Create a mock model to test
    class MockModelExceptionPredictor(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(3, 3))
            self.conv2 = torch.nn.Conv2d(64, 128, kernel_size=(3, 3))

        def forward(self, x):
            x = torch.relu(self.conv1(x))
            x = torch.relu(self.conv2(x))
            return x

    # Initialize the mock model
    model = MockModelExceptionPredictor()

    # Apply random shuffle weight
    try:
        model = random_shuffle_weight(model)
    except Exception as e:
        print(f"Exception Predicted: {str(e)}")
        assert isinstance(e, Exception), "Exception should be raised when trying to shuffle weights of a non-parametric model."
    else:
        print("Exception Not Predicted")

    print("Random Shuffle Exception Predictor Test Passed")