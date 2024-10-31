import os

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from torchvision import datasets, transforms

from models.UNet.model_unet import UNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

model = UNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

save_dir = './root'


def evaluate(model, data_loader, device):
    model.eval()
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            outputs_flat = outputs.permute(0, 2, 3, 1).contiguous().view(-1, 10)
            labels_flat = labels.repeat_interleave(28 * 28).view(-1)

            _, predicted = torch.max(outputs_flat, 1)
            total_correct += (predicted == labels_flat).sum().item()
            total_samples += labels_flat.size(0)

    accuracy = total_correct / total_samples
    return accuracy



# Training loop
num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)


        outputs_flat = outputs.permute(0, 2, 3, 1).contiguous().view(-1,10)
        labels_flat = labels.repeat_interleave(28 * 28).view(-1)


        labels_flat = labels_flat.long()

        loss = criterion(outputs_flat, labels_flat)
        loss.backward()
        optimizer.step()
    accuracy = evaluate(model, test_loader, device)
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}, Accuracy: {accuracy:.4f}')

    # Save the model
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    torch.save(model.state_dict(), os.path.join(save_dir, f'model_epoch_{epoch + 1}_acc_{accuracy:.4f}.pth'))
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

