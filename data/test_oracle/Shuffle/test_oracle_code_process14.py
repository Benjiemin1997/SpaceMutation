
from torch import nn

from mut.reverse_activation import reverse_activations


# Define a test case function for the reverse_activations method
def test_reverse_activations():
    # Create a simple model to test
    model = nn.Sequential(
        nn.Conv2d(3, 16, kernel_size=3),
        nn.ReLU(),
        nn.BatchNorm2d(16),
        nn.Conv2d(16, 32, kernel_size=3),
        nn.ReLU(inplace=True),  # Ensure inplace is True for comparison
        nn.BatchNorm2d(32),
    )
    
    # Apply reverse_activations method
    reversed_model = reverse_activations(model)
    
    # Assert that inplace attribute of ReLU layers was set to False
    for name, module in reversed_model.named_modules():
        if isinstance(module, nn.ReLU):
            assert not module.inplace, f"Inplace attribute of ReLU layer at '{name}' should be False"
            

    
    print("All tests passed for reverse_activations method.")

# Run the test case
test_reverse_activations()