from torch import nn
class LeNet5(nn.Sequential):
    def __init__(self):
        super(LeNet5, self).__init__(
            nn.Conv2d(1, 6, 5),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(256, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10),
            nn.ReLU()
        )

    def forward(self, x):
        return super(LeNet5, self).forward(x)


