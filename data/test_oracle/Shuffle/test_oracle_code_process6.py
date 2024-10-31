import torch
from torch import nn

from models.Shufflenetv2.shufflenetv2 import ShuffleNetV2
from mut.guassian_fuzz_splayers import gaussian_fuzzing_splayer
from mut.random_shuffle import random_shuffle_weight
from mut.remove_activation import remove_activations
from mut.replace_activation import replace_activations
from mut.uniform_fuzz import uniform_fuzz_weight

def test_model_after_mutation(model, std_ratio=0.8, target_layer_type=nn.Linear):
    # Test cases for original model
    input_data = torch.randn(1, 3, 224, 224)
    output_original = model(input_data)
    assert torch.all(output_original > -100), "Output is within reasonable bounds before mutation"

    # Mutation process
    mutated_model = gaussian_fuzzing_splayer(model, std_ratio, target_layer_type)
    mutated_model = random_shuffle_weight(mutated_model)
    mutated_model = remove_activations(mutated_model)
    mutated_model = replace_activations(mutated_model, 'relu')
    mutated_model = uniform_fuzz_weight(mutated_model)

    # Test cases for mutated model
    output_mutated = mutated_model(input_data)
    assert not torch.allclose(output_original, output_mutated), "Mutation has changed the output significantly"
    assert any(p.grad_fn is not None for p in mutated_model.parameters()), "At least one parameter should have a gradient function after mutation"
    
    print("All tests passed successfully.")

if __name__ == "__main__":
    # Initialize your model here
    model = ShuffleNetV2()
    test_model_after_mutation(model)
