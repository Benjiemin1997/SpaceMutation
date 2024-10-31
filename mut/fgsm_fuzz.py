import torch
from torch import nn

def fgsm_fuzz_weight(model, data_loader, epsilon=0.1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    for param in model.parameters():
        if param.requires_grad:
            param.grad = None

    criterion = nn.CrossEntropyLoss()
    for images, labels in data_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        model.zero_grad()
        loss.backward()
        with torch.no_grad():
            for param in model.parameters():
                if param.requires_grad:
                    param.data = param.data + epsilon * param.grad.sign()
    return model


def fgsm_fuzz_weight_mnist(model, data_loader, epsilon=0.1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # 清空梯度
    for param in model.parameters():
        if param.requires_grad:
            param.grad = None

    criterion = nn.CrossEntropyLoss()

    for images, labels in data_loader:
        images, labels = images.to(device), labels.to(device)
        assert labels.dim() == 1, f"Labels should be 1D, got {labels.dim()}D instead."
        outputs = model(images)
        outputs = outputs.mean(dim=(2, 3))
        assert outputs.shape == (images.shape[0], 10), "Output shape should be (batch_size, num_classes)."
        loss = criterion(outputs, labels)
        model.zero_grad()
        loss.backward()
        with torch.no_grad():
            for param in model.parameters():
                if param.requires_grad:
                    param.data.add_(epsilon * param.grad.sign())

    model.train()
    return model

