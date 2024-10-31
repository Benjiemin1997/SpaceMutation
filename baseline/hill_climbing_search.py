import random
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.transforms import functional as F
from models.AlexNet.model_alexnet import AlexNet
from util.seed_set import set_random_seed

from mut.fgsm_fuzz import fgsm_fuzz_weight
from mut.guassian_fuzz_splayers import gaussian_fuzzing_splayer
from mut.neuron_effect_blocks import neuron_effect_block
from mut.random_add_activation import add_activation
from mut.random_shuffle import random_shuffle_weight
from mut.remove_activation import remove_activations
from mut.replace_activation import replace_activations
from mut.reverse_activation import reverse_activations
from mut.uniform_fuzz import uniform_fuzz_weight
from util.composite_evaluate import composite_evaluate_model


def apply_mutation_by_type(model, dataloader, mutation_type):
    if mutation_type == 'structure':
        structure_mutation = random.choice([
            replace_activations,
            remove_activations,
            add_activation,
            reverse_activations,
            neuron_effect_block
        ])
        if neuron_effect_block == structure_mutation:
            model = structure_mutation(model, 0.1)
        else:
            model = structure_mutation(model)
    elif mutation_type == 'weights':
        weight_mutation = random.choice([
            fgsm_fuzz_weight,
            uniform_fuzz_weight,
            gaussian_fuzzing_splayer,
            random_shuffle_weight,
        ])
        if random_shuffle_weight == weight_mutation:
            model = weight_mutation(model)
        elif uniform_fuzz_weight == weight_mutation:
            model = weight_mutation(model, lower_bound=-0.1, upper_bound=0.1)
        elif fgsm_fuzz_weight == weight_mutation:
            model = weight_mutation(model, dataloader, epsilon=0.1)
        elif gaussian_fuzzing_splayer == weight_mutation:
            model = weight_mutation(model, std_ratio=0.5, target_layer_type=nn.Linear)
    else:
        model = None
    return model


def hill_climbing_search(start, dataloader, max_iterations, patience):
    current_model = start
    current_metrics = composite_evaluate_model(current_model, dataloader, 0.5, 0.1, 0.25, 1.0)
    current_accuracy = current_metrics[0]

    no_improvement_counter = 0

    for iteration in range(max_iterations):
        # Generate neighbors
        neighbor_model = apply_mutation_by_type(current_model, dataloader, random.choice(['structure', 'weights']))
        if neighbor_model is None:
            continue

        neighbor_metrics = composite_evaluate_model(neighbor_model, dataloader, 0.5, 0.1, 0.25, 1.0)
        neighbor_accuracy = neighbor_metrics[0]

        if neighbor_accuracy > current_accuracy:
            current_model = neighbor_model
            current_metrics = neighbor_metrics
            current_accuracy = neighbor_accuracy
            no_improvement_counter = 0
        else:
            no_improvement_counter += 1

        if no_improvement_counter >= patience:
            break

    return current_model, current_metrics


# Load data
batch_size = 64
data_transforms = {
    'test':
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
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
full_data_set = datasets.CIFAR100(root='.\data', train=False, download=True,
                                  transform=data_transforms['test'])
subset_length = int(1 * len(full_data_set))
subset_indices = list(range(subset_length))
data_sets = {'test': Subset(full_data_set, subset_indices)}
test_loader = DataLoader(data_sets['test'], batch_size=batch_size, shuffle=False, num_workers=0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = '/models/AlexNet/train_model/.pth'
model = AlexNet().to(device)
model.load_state_dict(torch.load(model_path), strict=False)

max_iterations = 10
patience = 10  

best_solution, best_metrics = hill_climbing_search(model, test_loader, max_iterations, patience)
torch.save(best_solution.state_dict(), 'AlexNet_Hill.pth')
print(composite_evaluate_model(best_solution,test_loader,0.5, 0.1, 0.25, 1.0))