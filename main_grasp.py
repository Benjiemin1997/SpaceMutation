import time
from torchvision.transforms import functional as F
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import random

from torch.utils.data import DataLoader, Subset
from torchvision import datasets

from llm_generate.prompt import modelPromptGenerator, get_mutant_content, mutatePromptGenerator, call_llms, prompt_1, \
    prompt_2, prompt_3, prompt_4, prompt_5, mutationimport, mutationImportPrompt, prompt_6, call_llms_ollama
from models.AlexNet.model_alexnet import AlexNet
from models.ResNet50.model_resnet50 import ResNet50
from models.VGG16.model_vgg16 import VGG16
from models.LeNet5.model_lenet5 import LeNet5
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


def construct_initial_solution(model, dataloader, mutations=[]):
    return apply_random_mutation(model, dataloader, mutations)


def local_search(model, dataloader, iterations, mutations=[]):
    best_model = model
    best_metrics = composite_evaluate_model(best_model, dataloader,0.5,0.1,0.25,1.0)
    best_accuracy, best_nbc, best_snac, best_nc = best_metrics

    for _ in range(iterations):
        new_model, new_mutations = apply_random_mutation(best_model, dataloader)
        if new_model is None:
            continue
        new_metrics = composite_evaluate_model(best_model, dataloader,0.5,0.1,0.25,1.0)
        new_accuracy, new_nbc, new_snac, new_nc = new_metrics

        if (new_nbc > best_nbc) and (new_snac > best_snac) and (new_nc > best_nc):
            best_model = new_model
            best_metrics = new_metrics
            mutations.extend(new_mutations)

    return best_model, mutations


def grasp(model, dataloader, num_iterations, local_search_iterations):
    print("Starting GRASP search...")
    best_solution = model
    best_solution_metrics = composite_evaluate_model(best_solution, dataloader,0.5,0.1,0.25,1.0)
    best_solution_accuracy, best_solution_nbc, best_solution_snac, best_solution_nc = best_solution_metrics
    mutations = []
    for _ in range(num_iterations):
        # Construct initial solution
        initial_solution, initial_mutations = construct_initial_solution(model, dataloader, mutations)
        mutations.extend(initial_mutations)

        # Perform local search
        improved_solution, local_search_mutations = local_search(initial_solution, dataloader, local_search_iterations)
        mutations.extend(local_search_mutations)

        # Update global best solution
        improved_solution_metrics = composite_evaluate_model(improved_solution, dataloader,0.5,0.1,0.25,1.0)
        improved_solution_accuracy, improved_solution_nbc, improved_solution_snac, improved_solution_nc = improved_solution_metrics

        if (improved_solution_nbc > best_solution_nbc) and (
                improved_solution_snac > best_solution_snac) and (improved_solution_nc > best_solution_nc):
            best_solution = improved_solution
            best_solution_metrics = improved_solution_metrics
    return best_solution, mutations


def apply_random_mutation(model, dataloader, mutations=[]):
    mutation_type1 = random.choice(['structure', 'weights'])
    mutation_type2 = random.choice(['structure', 'weights'])
    model, mutation_info = apply_mutation_by_type(model, dataloader, mutation_type1)
    if model is not None:
        mutations.append(mutation_info)
    model, mutation_info = apply_mutation_by_type(model, dataloader, mutation_type2)
    if model is not None:
        mutations.append(mutation_info)

    return model, mutations

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
        mutation_info = (mutation_type, structure_mutation.__name__)
    elif mutation_type == 'weights':
        weight_mutation = random.choice([
            fgsm_fuzz_weight,
            uniform_fuzz_weight,
            gaussian_fuzzing_splayer,
            random_shuffle_weight,
        ])
        if random_shuffle_weight == weight_mutation:
            model = weight_mutation(model)
            mutation_info = (mutation_type, weight_mutation.__name__)
        elif uniform_fuzz_weight == weight_mutation:
            model = weight_mutation(model, lower_bound=-0.1, upper_bound=0.1)
            mutation_info = (mutation_type, weight_mutation.__name__, -0.1, 0.1)
        elif fgsm_fuzz_weight == weight_mutation:
            model = weight_mutation(model, dataloader, epsilon=0.1)
            mutation_info = (mutation_type, weight_mutation.__name__, 0.1)
        elif gaussian_fuzzing_splayer == weight_mutation:
            model = weight_mutation(model, std_ratio=0.5, target_layer_type=nn.Linear)
            mutation_info = (mutation_type, weight_mutation.__name__, 0.5, nn.Linear)
    else:
        model = None
        mutation_info = None
    return model, mutation_info

if __name__ == "__main__":
    # Set random seed for reproducibility
    seed = 10000
    set_random_seed(seed)
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
                        transforms.RandomHorizontalFlip(p=0.5),  # 水平翻转
                        transforms.Lambda(lambda img: img + torch.randn_like(img) * 0.1),  # 噪声扰动
                        transforms.Lambda(lambda img: F.adjust_contrast(img, contrast_factor=1.5))
                    ]

                )
            ]),
    }
    # Load the full dataset
    full_data_set = datasets.CIFAR100(root='D://pyproject//NetMut//models//AlexNet//data', train=False, download=True,
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
    model_path = './models/ResNet50/train_model/model_epoch_20_acc_97.5020.pth'
    model = ResNet50().to(device)
    model.load_state_dict(torch.load(model_path), strict=False)
    start_time = time.time()
    # Adaptive search and multi-level optimization
    optimized_model, mutations = grasp(model, test_loader, num_iterations=5, local_search_iterations=5)
    end_time = time.time()
    execution_time = time.time() - start_time
    torch.save(optimized_model.state_dict(), 'ResNet50_SM.pth')
    print(f'Time taken to optimize the model: {execution_time:.2f} seconds')
    print("The total number of mutation operations", len(mutations))
    mutation_model_prompt = modelPromptGenerator(optimized_model)
    mutation_operations = [mutation[1] for mutation in mutations]
    file_mapping = {
            "uniform_fuzz_weight": "uniform_fuzz.py",
            "gaussian_fuzzing_splayer": "guassian_fuzz_splayers.py",
            "random_shuffle_weight": "random_shuffle.py",
            "remove_activations": "remove_activation.py",
            "fgsm_fuzz_weight": "fgsm_fuzz.py",
            "replace_activations": "replace_activation.py",
            "neuron_effect_block": "neuron_effect_blocks.py",
            "add_activation": "random_add_activation.py",
            "reverse_activations": "reverse_activation.py"
        }
    mut_operator = list(set(mutation_operations))
    second_elements = [file_mapping[mutation] for mutation in mut_operator]
    mutant_paths = []
    for element in second_elements:
        if not element.startswith('./mut/'):
            element = './mut/' + element
        mutant_paths.append(element)
    mutant_content = ''
    for mutant_path in mutant_paths:
        mutant_content = get_mutant_content(mutant_path)
    mutant_function_prompt = mutatePromptGenerator(mutant_content)
    mutant_mutationimport = mutationImportPrompt(mutationimport)

    model_name = "qwen2-7b-instruct"
    temperature = 0.7
    messages = [{"role": "system", "content": prompt_1},
                {"role": "system", "content": prompt_2},
                {"role": "system", "content": prompt_3 + '/n' + mutant_function_prompt},
                {"role": "system", "content": prompt_4 + '/n' + mutation_model_prompt},
                {"role": "system", "content": prompt_6 + '/n' + mutant_mutationimport},
                {"role": "user", "content": prompt_5}
                ]
    # Call the LLM and get the generated code
    start_time = time.time()
    generated_code = call_llms(model_name, messages, temperature)
    execution_time = time.time() - start_time
    print(f'Time taken to generate code: {execution_time:.2f} seconds')

    # Write the generated code to the output file
    output_path = f'./data/test_oracle/test_oracle_code.py'
    with open(output_path, mode='w', encoding='utf-8') as output_file:
        output_file.write(generated_code)
        print("The test oracle has been generated")

    #Evaluate the optimized model
    test_loader = DataLoader(data_sets['test'], batch_size=batch_size, shuffle=False, num_workers=0)
    start_time = time.time()
    accuracy, nbc, snac, nc = composite_evaluate_model(optimized_model, test_loader,1.0,0.1,0.25,1.0)
    execution_time = time.time() - start_time
    print(f'Time taken to testing  : {execution_time:.2f} seconds')
    print(f'NBC after optimization: {nbc:.2f}')
    print(f'SNAC after optimization: {snac:.2f}')
    print(f'NC after optimization: {nc:.2f}')
