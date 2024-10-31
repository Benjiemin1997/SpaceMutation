import random

import torch
from torch.utils.data import Subset, DataLoader
from torchvision import transforms, datasets
from torchvision.transforms import functional as F

from models.AlexNet.model_alexnet import AlexNet
from models.VGG16.model_vgg16 import VGG16
from util.composite_evaluate import composite_evaluate_model
from util.seed_set import set_random_seed

#seed = 10000
#set_random_seed(seed)
# Load data
batch_size = 64
data_transforms = {
    'test':
        transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            random.choice([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.Lambda(lambda img: img + torch.randn_like(img) * 0.1),
                transforms.Lambda(lambda img: F.adjust_contrast(img, contrast_factor=1.5))
            ])
        ]),
}

# Load the full dataset
full_data_set = datasets.CIFAR100(root='D://pyproject//NetMut//models//AlexNet//data', train=False, download=True,
                                  transform=data_transforms['test'])
subset_length = int(0.002 * len(full_data_set))
subset_indices = list(range(subset_length))
data_sets = {'test': Subset(full_data_set, subset_indices)}
test_loader = DataLoader(data_sets['test'], batch_size=batch_size, shuffle=False, num_workers=0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Load pre-trained model
new_model = VGG16().to(device)
new_model.load_state_dict(torch.load('D://pyproject//NetMut//baseline//VGG16//VGG16_GA.pth'), strict=False)
new_model.eval()
print(composite_evaluate_model(new_model,test_loader,0.5, 0.1, 0.25, 1.0))