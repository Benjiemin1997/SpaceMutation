from torch import nn
from torchvision.models import resnet50


class ResNet50(nn.Module):
    def __init__(self, num_classes=100):
        super(ResNet50, self).__init__()
        self.resnet = resnet50(pretrained=True)
        self.resnet.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.resnet.fc.in_features, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.resnet(x)
        return x

