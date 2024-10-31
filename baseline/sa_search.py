import math
import random

from torch.utils.data import Subset, DataLoader
from torchvision.transforms import functional as F
import torch
from torchvision import transforms, datasets

from models.AlexNet.model_alexnet import AlexNet
from util.apply_mutation import apply_mutation_by_type
from util.composite_evaluate import composite_evaluate_model




def construct_initial_solution(model, dataloader, mutations=[]):
    return apply_random_mutation(model, dataloader, mutations)


def apply_random_mutation(model, dataloader, mutations=[]):
    mutation_type1 = random.choice(['structure', 'weights'])
    mutation_type2 = random.choice(['structure', 'weights'])

    model= apply_mutation_by_type(model, dataloader, mutation_type1)
    model = apply_mutation_by_type(model, dataloader, mutation_type2)
    return model, mutations


def simulated_annealing(model, dataloader, initial_temperature, cooling_rate, max_iterations):
    current_model = model
    current_metrics = composite_evaluate_model(current_model, dataloader, 0.5, 0.1, 0.25, 1.0)
    current_accuracy, current_nbc, current_snac, current_nc = current_metrics
    best_model = current_model
    best_metrics = current_metrics
    temperature = initial_temperature

    for _ in range(max_iterations):
        new_model, _ = apply_random_mutation(current_model, dataloader)
        if new_model is None:
            continue

        new_metrics = composite_evaluate_model(new_model, dataloader, 0.5, 0.1, 0.25, 1.0)
        new_accuracy, new_nbc, new_snac, new_nc = new_metrics

        # 计算能量差
        delta_accuracy = new_accuracy - current_accuracy
        acceptance_probability = math.exp(-delta_accuracy / temperature)

        # 决定是否接受新模型
        if delta_accuracy > 0 or random.random() < acceptance_probability:
            current_model = new_model
            current_metrics = new_metrics
            current_accuracy, current_nbc, current_snac, current_nc = current_metrics
            if current_accuracy < best_metrics[0]:
                best_model = current_model
                best_metrics = current_metrics
        temperature *= cooling_rate

    return best_model, best_metrics

initial_temperature = 1.0
cooling_rate = 0.99
max_iterations = 10

# Load data
batch_size = 64
data_transforms = {
        'test':
            transforms.Compose([
                transforms.Resize(256),  # 将图像大小调整为256x256
                transforms.CenterCrop(224),  # 中心裁剪为224x224
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                random.choice(
                    [
                        transforms.RandomHorizontalFlip(p=0.5),  # 水平翻转
                        transforms.Lambda(lambda img: img + torch.randn_like(img) * 0.1),  # 噪声扰动
                        transforms.Lambda(lambda img: F.adjust_contrast(img, contrast_factor=1.5))
                    ]

                )
            ]),
    }

# Load the full dataset
full_data_set = datasets.CIFAR100(root='.\data', train=False, download=True,
                                      transform=data_transforms['test'])

# Get dataset length
dataset_length = len(full_data_set)
subset_length = int(1 * dataset_length)
subset_indices = list(range(subset_length))
# Create a subset dataset
data_sets = {'test': Subset(full_data_set, subset_indices)}
test_loader = DataLoader(data_sets['test'], batch_size=batch_size, shuffle=False, num_workers=0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Load pre-trained model
model_path = '/models/AlexNet/train_model/.pth'
model = AlexNet().to(device)
model.load_state_dict(torch.load(model_path), strict=False)
best_model, best_metrics = simulated_annealing(model, test_loader, initial_temperature, cooling_rate, max_iterations)
print(composite_evaluate_model(best_model,test_loader,0.5, 0.1, 0.25, 1.0))