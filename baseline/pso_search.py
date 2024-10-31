import random

import torch
from torch import nn
from torch.utils.data import Subset, DataLoader
from torchvision import transforms, datasets
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


class Particle:
    def __init__(self, model, dataloader):
        self.position = model
        # Initialize velocity as zero (or a model-like structure with zero weights)
        self.velocity = self.initialize_velocity(model)
        self.best_position = model
        self.best_score = float('inf')
        self.dataloader = dataloader

    def initialize_velocity(self, model):
        velocity = {}
        for name, param in model.named_parameters():
            velocity[name] = torch.zeros_like(param)
        return velocity

    def evaluate(self):
        metrics = composite_evaluate_model(self.position, self.dataloader, 0.5, 0.1, 0.25, 1.0)
        accuracy, nbc, snac, nc = metrics
        score = self.calculate_score(accuracy, nbc, snac, nc)
        if score < self.best_score:
            self.best_score = score
            self.best_position = self.position
        return score

    def calculate_score(self, accuracy, nbc, snac, nc):
        return accuracy - nbc - snac - nc

    def update_velocity(self, global_best, inertia=0.5, cognitive=1.0, social=1.0):
        for name, param in self.position.named_parameters():
            cognitive_component = cognitive * random.random() * (self.best_position.state_dict()[name] - param)
            social_component = social * random.random() * (global_best.state_dict()[name] - param)
            self.velocity[name] = inertia * self.velocity[name] + cognitive_component + social_component

    def update_position(self):
        with torch.no_grad():
            for name, param in self.position.named_parameters():
                param.add_(self.velocity[name])  # Update the model parameters using velocity
def apply_velocity(model):
    mutation_type = random.choice(['structure', 'weights'])
    model, _ = apply_mutation_by_type(model, None, mutation_type)
    return model
def pso(model, dataloader, num_particles, iterations):
    particles = [Particle(model, dataloader) for _ in range(num_particles)]
    global_best_position = model
    global_best_score = float('inf')

    for _ in range(iterations):
        for particle in particles:
            score = particle.evaluate()
            if score < global_best_score:
                global_best_score = score
                global_best_position = particle.position

        for particle in particles:
            particle.update_velocity(global_best_position)
            particle.update_position()

    return global_best_position
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
            model = structure_mutation(model,0.1)
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

batch_size = 64
data_transforms = {
        'test':
            transforms.Compose([
                transforms.Resize(256), 
                transforms.CenterCrop(224), 
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                random.choice(
                    [
                        transforms.RandomHorizontalFlip(p=0.5), 
                        transforms.Lambda(lambda img: img + torch.randn_like(img) * 0.1),  
                        transforms.Lambda(lambda img: F.adjust_contrast(img, contrast_factor=1.5))
                    ]

                )
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
best_model = pso(model, test_loader, num_particles=5, iterations=5)
torch.save(best_model.state_dict(), 'AlexNet_PSO.pth')
print(composite_evaluate_model(best_model,test_loader,0.5, 0.1, 0.25, 1.0))