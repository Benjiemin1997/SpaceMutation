import random

from torch import nn
from mut.fgsm_fuzz import fgsm_fuzz_weight, fgsm_fuzz_weight_mnist
from mut.guassian_fuzz_splayers import gaussian_fuzzing_splayer
from mut.neuron_effect_blocks import neuron_effect_block
from mut.random_add_activation import add_activation
from mut.random_shuffle import random_shuffle_weight
from mut.remove_activation import remove_activations
from mut.replace_activation import replace_activations
from mut.reverse_activation import reverse_activations
from mut.uniform_fuzz import uniform_fuzz_weight

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

def apply_mutation_by_type_mnist(model, dataloader, mutation_type):
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
            fgsm_fuzz_weight_mnist,
            uniform_fuzz_weight,
            gaussian_fuzzing_splayer,
            random_shuffle_weight,
        ])
        if random_shuffle_weight == weight_mutation:
            model = weight_mutation(model)

        elif uniform_fuzz_weight == weight_mutation:
            model = weight_mutation(model, lower_bound=-0.1, upper_bound=0.1)

        elif fgsm_fuzz_weight_mnist == weight_mutation:
            model = weight_mutation(model, dataloader, epsilon=0.1)

        elif gaussian_fuzzing_splayer == weight_mutation:
            model = weight_mutation(model, std_ratio=0.5, target_layer_type=nn.Linear)

    else:
        model = None
    return model