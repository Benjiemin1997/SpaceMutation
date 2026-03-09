import os
import sys
import torch

from example.model_unet import UNet

PROJECT_ROOT = os.environ.get("PROJECT_ROOT", "/example")
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def load_unet_model():
    model_path = os.environ.get("MODEL_PATH", "/example/UNet_SM_knee.pth")
    device = torch.device("cpu")

    model = UNet(in_channels=3, num_classes=21)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    return model