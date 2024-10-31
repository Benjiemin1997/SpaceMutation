import random
import numpy as np
import torch
from torch import nn
from torch.utils.data import Subset, DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import functional as F
from models.AlexNet.model_alexnet import AlexNet
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
from util.seed_set import set_random_seed


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


def construct_initial_solution(model, dataloader, pheromone_matrix):
    mutation_type = select_mutation_based_on_pheromones(pheromone_matrix)
    mutated_model = apply_mutation_by_type(model, dataloader, mutation_type)
    return mutated_model, [(mutation_type, None)]


def select_mutation_based_on_pheromones(pheromone_matrix):
    total_pheromone = sum(pheromone_matrix.values())
    probabilities = {mt: p / total_pheromone for mt, p in pheromone_matrix.items()}
    return np.random.choice(list(probabilities.keys()), p=list(probabilities.values()))


def update_pheromones(pheromone_matrix, ants_solutions, evaporation_rate, q,dataloader):
    for mutation_type in pheromone_matrix:
        pheromone_matrix[mutation_type] *= (1 - evaporation_rate)

    for sol, muts in ants_solutions:
        for mut_type, _ in muts:
            if mut_type in pheromone_matrix:
                pheromone_matrix[mut_type] += q / composite_evaluate_model(sol, dataloader, 0.5, 0.1, 0.25, 1.0)[0]

def apply_random_mutation(model, dataloader):
    mutation_type1 = random.choice(['structure', 'weights'])
    mutation_type2 = random.choice(['structure', 'weights'])
    model = apply_mutation_by_type(model, dataloader, mutation_type1)
    model = apply_mutation_by_type(model, dataloader, mutation_type2)

    return model

def local_search(model, dataloader, iterations):
    best_model = model
    best_metrics = composite_evaluate_model(best_model, dataloader, 0.5, 0.1, 0.25, 1.0)
    best_accuracy, _, _, _ = best_metrics

    for _ in range(iterations):
        new_model = apply_random_mutation(best_model, dataloader)
        if new_model is None:
            continue
        new_metrics = composite_evaluate_model(new_model, dataloader, 0.5, 0.1, 0.25, 1.0)
        new_accuracy, _, _, _ = new_metrics

        if new_accuracy < best_accuracy:
            best_model = new_model
            best_metrics = new_metrics

    return best_model, []


def ant_colony_optimization(model, dataloader, num_ants, num_iterations, evaporation_rate, q, local_search_iterations):
    mutation_types = ['structure', 'weights']
    pheromone_matrix = {mt: 1.0 for mt in mutation_types}

    best_solution = model
    best_solution_metrics = composite_evaluate_model(best_solution, dataloader, 0.5, 0.1, 0.25, 1.0)
    best_solution_accuracy, _, _, _ = best_solution_metrics

    for _ in range(num_iterations):
        ants_solutions = [construct_initial_solution(model, dataloader, pheromone_matrix) for _ in range(num_ants)]
        ants_solutions = [local_search(sol, dataloader, local_search_iterations) for sol, _ in ants_solutions]
        for sol, _ in ants_solutions:
            metrics = composite_evaluate_model(sol, dataloader, 0.5, 0.1, 0.25, 1.0)
            accuracy, _, _, _ = metrics

            if accuracy < best_solution_accuracy:
                best_solution = sol
                best_solution_metrics = metrics

        # Update pheromone matrix
        update_pheromones(pheromone_matrix, ants_solutions, evaporation_rate, q, dataloader)

    return best_solution, pheromone_matrix



num_ants = 5
num_iterations = 10
evaporation_rate = 0.5
q = 10
local_search_iterations = 5

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

full_data_set = datasets.CIFAR100(root='.\data', train=False, download=True,
                                  transform=data_transforms['test'])
dataset_length = len(full_data_set)
subset_length = int(1 * dataset_length)
subset_indices = list(range(subset_length))
data_sets = {'test': Subset(full_data_set, subset_indices)}
test_loader = DataLoader(data_sets['test'], batch_size=batch_size, shuffle=False, num_workers=0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = '/models/AlexNet/train_model/.pth'
model = AlexNet().to(device)
model.load_state_dict(torch.load(model_path), strict=False)

best_solution, pheromone_matrix = ant_colony_optimization(model, test_loader, num_ants, num_iterations, evaporation_rate,
                                                          q, local_search_iterations)
print(composite_evaluate_model(best_solution,test_loader,0.5, 0.1, 0.25, 1.0))