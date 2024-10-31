import random
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


def initialize_particles(num_particles, model, dataloader):
    particles = []
    for _ in range(num_particles):
        particle = {
            'position': clone_model(model),
            'fitness': evaluate_particle_fitness({'position': clone_model(model)}, dataloader)
        }
        particles.append(particle)
    return particles


def clone_model(model):
    cloned_model = type(model)()
    cloned_model.load_state_dict(model.state_dict())
    return cloned_model.to(device)


def crossover(parent1, parent2):
    alpha = random.uniform(0, 1)
    child = clone_model(parent1)
    for p1, p2, c in zip(parent1.parameters(), parent2.parameters(), child.parameters()):
        c.data.copy_(alpha * p1.data + (1 - alpha) * p2.data)
    return child


def selection(population, num_parents):
    population.sort(key=lambda x: x['fitness'], reverse=True)
    parents = population[:num_parents]
    return parents

def evolve_population(population, dataloader, num_offspring, num_parents):
    parents = selection(population, num_parents)

    offspring = []
    while len(offspring) < num_offspring:
        parent1, parent2 = random.choices(parents, k=2)

        # Perform crossover
        child = crossover(parent1['position'], parent2['position'])

        # Apply random mutation
        child = apply_random_mutation(child, dataloader)

        # Evaluate the fitness of the new individual
        fitness = evaluate_particle_fitness({'position': child}, dataloader)
        offspring.append({'position': child, 'fitness': fitness})

    # Merge parents and offspring
    population.extend(offspring)
    return population


def genetic_algorithm(model, dataloader, num_generations, population_size, num_parents, num_offspring):
    print("Starting Genetic Algorithm...")
    population = initialize_particles(population_size, model, dataloader)

    for generation in range(num_generations):
        print(f"Generation {generation + 1}/{num_generations}")
        population = evolve_population(population, dataloader, num_offspring, num_parents)

    population.sort(key=lambda x: x['fitness'], reverse=True)
    best_solution = population[0]['position']
    best_fitness = population[0]['fitness']

    return best_solution, best_fitness


def apply_random_mutation(model, dataloader):
    mutation_type1 = random.choice(['structure', 'weights'])
    mutation_type2 = random.choice(['structure', 'weights'])

    # Apply first mutation
    print("Applying first mutation")
    model = apply_mutation_by_type(model, dataloader, mutation_type1)

    # Apply second mutation
    print("Applying second mutation")
    model = apply_mutation_by_type(model, dataloader, mutation_type2)

    return model


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


def evaluate_particle_fitness(particle, dataloader):
    metrics = composite_evaluate_model(particle['position'], dataloader, 0.5, 0.1, 0.25, 1.0)
    fitness = metrics[0] 
    return fitness


seed = 10000
set_random_seed(seed)

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
full_data_set = datasets.CIFAR100(root='D://pyproject//NetMut//models//AlexNet//data', train=False, download=True,
                                  transform=data_transforms['test'])
num_particles = 10 
num_generations = 5
num_parents = 2 
num_offspring = 5  

# Get dataset length
dataset_length = len(full_data_set)
subset_length = int(1 * dataset_length)
subset_indices = list(range(subset_length))
# Create a subset dataset
data_sets = {'test': Subset(full_data_set, subset_indices)}
test_loader = DataLoader(data_sets['test'], batch_size=batch_size, shuffle=False, num_workers=0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Load pre-trained model
model_path = '/models/AlexNet/train_model/model_epoch_30_acc_97.7960.pth'
model = AlexNet().to(device)
model.load_state_dict(torch.load(model_path), strict=False)

best_solution, best_fitness = genetic_algorithm(model, test_loader, num_generations, num_particles, num_parents,
                                                num_offspring)
torch.save(best_solution.state_dict(), 'AlexNet_GA.pth')
print(composite_evaluate_model(best_solution,test_loader,0.5, 0.1, 0.25, 1.0))