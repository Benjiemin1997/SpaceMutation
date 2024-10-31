import heapq
import random
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
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

def heuristic_cost_estimate(node, goal,dataloader):
    node_metrics = composite_evaluate_model(node.state, dataloader, 0.5, 0.1, 0.25, 1.0)
    goal_metrics = goal
    heuristic = abs(goal_metrics[0] - node_metrics[0])  # 以准确率为基准
    return heuristic

def actual_cost_to_reach(node,dataloader):
    parent_metrics = composite_evaluate_model(node.parent.state, dataloader, 0.5, 0.1, 0.25, 1.0) if node.parent else 0
    node_metrics = composite_evaluate_model(node.state, dataloader, 0.5, 0.1, 0.25, 1.0)
    cost = parent_metrics[0] - node_metrics[0]
    return cost

class Node:
    def __init__(self, state, parent=None, cost=0, heuristic=0):
        self.state = state
        self.parent = parent
        self.cost = cost
        self.heuristic = heuristic

    def __lt__(self, other):
        return (self.cost + self.heuristic) < (other.cost + other.heuristic)

def a_star_search(start, dataloader, goal, max_expansions):
    open_list = []
    heapq.heappush(open_list, Node(start))

    closed_list = set()

    while open_list and len(closed_list) < max_expansions:
        current_node = heapq.heappop(open_list)
        current_state = current_node.state

        if current_state is None:
            continue

        current_metrics = composite_evaluate_model(current_state, dataloader, 0.5, 0.1, 0.25, 1.0)
        if current_metrics[0] <= goal[0]:
            return current_state, current_metrics

        closed_list.add(current_state)

        for mutation_type in ['structure', 'weights']:
            child_state = apply_mutation_by_type(current_state, dataloader, mutation_type)
            if child_state is None:
                continue

            cost = actual_cost_to_reach(Node(child_state, current_node),dataloader)
            heuristic = heuristic_cost_estimate(Node(child_state, current_node), goal,dataloader)
            child_node = Node(child_state, current_node, cost, heuristic)

            if child_state not in closed_list:
                heapq.heappush(open_list, child_node)

    return start, composite_evaluate_model(start, dataloader, 0.5, 0.1, 0.25, 1.0)

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
full_data_set = datasets.CIFAR100(root='./data', train=False, download=True,
                                  transform=data_transforms['test'])
subset_length = int(1 * len(full_data_set))
subset_indices = list(range(subset_length))
data_sets = {'test': Subset(full_data_set, subset_indices)}
test_loader = DataLoader(data_sets['test'], batch_size=batch_size, shuffle=False, num_workers=0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = '/models/AlexNet/train_model/.pth'
model = AlexNet().to(device)
model.load_state_dict(torch.load(model_path), strict=False)

max_expansions = 10
goal = composite_evaluate_model(model, test_loader, 0.5, 0.1, 0.25, 1.0)
best_solution, best_metrics = a_star_search(model, test_loader, goal, max_expansions)
print(composite_evaluate_model(best_solution,test_loader,0.5, 0.1, 0.25, 1.0))