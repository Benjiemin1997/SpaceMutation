import torch
from torch import nn
from mut.guassian_fuzz_splayers import gaussian_fuzzing_splayer
from mut.random_shuffle import random_shuffle_weight
from mut.remove_activation import remove_activations
from mut.replace_activation import replace_activations
from mut.uniform_fuzz import uniform_fuzz_weight

def reverse_activations(model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for name, module in model.named_children():
        if isinstance(module, nn.ReLU):
            module.inplace = False
            prev_module = list(model.named_children())[list(model.named_children()).index((name, module)) - 1][1]
            prev_module.register_forward_hook(lambda module, input, output: -output)
    return model

def test_reverse_activations():
    # Create a simple test model
    class TestModel(nn.Module):
        def __init__(self):
            super(TestModel, self).__init__()
            self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
            self.relu1 = nn.ReLU()
            self.maxpool1 = nn.MaxPool2d(kernel_size=2)
            self.fc1 = nn.Linear(256, 120)
            self.fc2 = nn.Linear(120, 84)
            self.prelu1 = nn.PReLU()
            self.fc3 = nn.Linear(84, 10)

        def forward(self, x):
            x = self.conv1(x)
            x = self.relu1(x)
            x = self.maxpool1(x)
            x = x.view(-1, 256)
            x = self.fc1(x)
            x = self.fc2(x)
            x = self.prelu1(x)
            x = self.fc3(x)
            return x

    # Initialize the model
    model = TestModel()

    # Reverse activations
    reversed_model = reverse_activations(model)

    # Fuzzing test cases
    test_cases = [
        gaussian_fuzzing_splayer(reversed_model),
        random_shuffle_weight(reversed_model),
        remove_activations(reversed_model),
        replace_activations(reversed_model),
        uniform_fuzz_weight(reversed_model)
    ]

    for test_case in test_cases:
        input_data = torch.randn(1, 1, 32, 32)
        original_output = model(input_data)
        fuzzed_output = test_case(input_data)
        assert torch.allclose(original_output, -fuzzed_output), f"Fuzzing did not work as expected. Original output: {original_output}, Fuzzed output: {fuzzed_output}"