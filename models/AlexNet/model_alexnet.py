from torch import nn
from torchvision.models import alexnet



class AlexNet(nn.Module):
    def __init__(self, num_classes=100):
        super(AlexNet, self).__init__()
        from torchvision.models import AlexNet_Weights
        self.alexnet = alexnet(weights=AlexNet_Weights.IMAGENET1K_V1)

        # 修改分类器部分
        self.alexnet.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.alexnet(x)
        return x
