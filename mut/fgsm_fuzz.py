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

        # 确保 labels 的形状是 (batch_size,)
        assert labels.dim() == 1, f"Labels should be 1D, got {labels.dim()}D instead."

        # 前向传播
        outputs = model(images)

        # 将输出张量从 (batch_size, num_classes, H, W) 转换为 (batch_size, num_classes)
        # 这里假设 H 和 W 相等，并且都是 1（因为 MNIST 图像经过 UNet 处理后可能被压缩）
        outputs = outputs.mean(dim=(2, 3))  # 将 H 和 W 维度平均

        # 确保模型输出的形状正确
        assert outputs.shape == (images.shape[0], 10), "Output shape should be (batch_size, num_classes)."

        # 计算损失
        loss = criterion(outputs, labels)

        # 反向传播
        model.zero_grad()
        loss.backward()

        # 对权重进行 FGSM 扰动
        with torch.no_grad():
            for param in model.parameters():
                if param.requires_grad:
                    param.data.add_(epsilon * param.grad.sign())

    model.train()
    return model

