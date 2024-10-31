import torch
import torch.nn as nn


def nbc_compute(activations, high_n, low_n, lower_corner_neurons, total_neurons, upper_corner_neurons):
    for activation in activations:
        upper_corner_neurons += (activation > high_n).sum().item()
        lower_corner_neurons += (activation < low_n).sum().item()
        total_neurons += activation.numel()
    return lower_corner_neurons, total_neurons, upper_corner_neurons


def hook_fn(module, input, output):
    if hasattr(module, 'output'):
        module.output = output
    else:
        setattr(module, 'output', output)


def snc_compute(device, high_n, inputs, model, total_neurons, upper_corner_neurons):
    activations = get_activations(model, inputs, device)
    for activation in activations:
        upper_corner_neurons += (activation > high_n).sum().item()
        total_neurons += activation.numel()
    return total_neurons, upper_corner_neurons


def nc_compute(inputs, model, neuron_activated, threshold):
    model(inputs)
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
            layer_outputs = getattr(module, 'output', None)
            if layer_outputs is not None:
                if isinstance(module, nn.Linear):
                    activated_neurons = (layer_outputs > threshold).any(dim=0)
                elif isinstance(module, nn.Conv2d):
                    layer_outputs_flat = layer_outputs.view(layer_outputs.size(0), -1)
                    activated_neurons = (layer_outputs_flat > threshold).any(dim=0)
                    activated_neurons = activated_neurons.view(-1)[:module.out_channels]
                neuron_activated[name][activated_neurons] = 1


def get_activations(model, inputs, device):
    activations = []

    def hook_fn(module, input, output):
        activations.append(output)

    hooks = []
    for layer in model.modules():
        if isinstance(layer, (nn.Conv2d, nn.Linear)):
            hooks.append(layer.register_forward_hook(hook_fn))
    model(inputs.to(device))
    for hook in hooks:
        hook.remove()
    return activations


def composite_evaluate_model(model, data_loader, high_n, low_n, threshold, high_n_s):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    correct = 0
    total = 0
    upper_corner_neurons = 0
    lower_corner_neurons = 0
    neuron_activated = {}
    total_neurons = 0
    total_neurons_nc = 0
    upper_corner_neurons_snc = 0
    total_neurons_snc = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            neuron_activated[name] = torch.zeros(module.out_features).to(device)
            total_neurons_nc += module.out_features
        elif isinstance(module, nn.Conv2d):
            neuron_activated[name] = torch.zeros(module.out_channels).to(device)
            total_neurons_nc += module.out_channels
            module.register_forward_hook(hook_fn)

    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)

            nc_compute(images, model, neuron_activated, threshold)
            activations = get_activations(model, images, device)

            lower_corner_neurons, total_neurons, upper_corner_neurons = nbc_compute(activations, high_n, low_n,
                                                                                    lower_corner_neurons, total_neurons,
                                                                                    upper_corner_neurons)
            total_neurons_snc, upper_corner_neurons_snc = snc_compute(device, high_n_s, images, model,
                                                                      total_neurons_snc,
                                                                      upper_corner_neurons_snc)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    if total_neurons == 0:
        return 0.0
    nbc = (upper_corner_neurons + lower_corner_neurons) / (2 * total_neurons) * 100
    activated_neurons = sum([torch.sum(v) for v in neuron_activated.values()])
    coverage = (activated_neurons.item() / total_neurons_nc) * 100
    if total_neurons == 0:
        return 0.0
    snac = (upper_corner_neurons_snc / total_neurons_snc) * 100
    return accuracy, nbc, snac, coverage


def composite_evaluate_model_mnist(model, data_loader, high_n, low_n, threshold, high_n_s):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    correct = 0
    total = 0
    upper_corner_neurons = 0
    lower_corner_neurons = 0
    neuron_activated = {}
    total_neurons = 0
    total_neurons_nc = 0
    upper_corner_neurons_snc = 0
    total_neurons_snc = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            neuron_activated[name] = torch.zeros(module.out_features).to(device)
            total_neurons_nc += module.out_features
        elif isinstance(module, nn.Conv2d):
            neuron_activated[name] = torch.zeros(module.out_channels).to(device)
            total_neurons_nc += module.out_channels
            module.register_forward_hook(hook_fn)
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            outputs_flat = outputs.permute(0, 2, 3, 1).contiguous().view(-1, 10)
            labels_flat = labels.repeat_interleave(28 * 28).view(-1)

            nc_compute(images, model, neuron_activated, threshold)
            activations = get_activations(model, images, device)

            lower_corner_neurons, total_neurons, upper_corner_neurons = nbc_compute(activations, high_n, low_n,
                                                                                    lower_corner_neurons, total_neurons,
                                                                                    upper_corner_neurons)
            total_neurons_snc, upper_corner_neurons_snc = snc_compute(device, high_n_s, images, model,
                                                                      total_neurons_snc,
                                                                      upper_corner_neurons_snc)
            _, predicted = torch.max(outputs_flat, 1)
            correct += (predicted == labels_flat).sum().item()
            total += labels_flat.size(0)
    accuracy = 100 * correct / total
    if total_neurons == 0:
        return 0.0
    nbc = (upper_corner_neurons + lower_corner_neurons) / (2 * total_neurons) * 100
    activated_neurons = sum([torch.sum(v) for v in neuron_activated.values()])
    coverage = (activated_neurons.item() / total_neurons_nc) * 100
    if total_neurons == 0:
        return 0.0
    snac = (upper_corner_neurons_snc / total_neurons_snc) * 100
    return accuracy, nbc, snac, coverage
