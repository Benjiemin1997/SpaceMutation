import torch
def random_shuffle_weight(model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    def shuffle_weights(weights):
        weight_shape = weights.size()
        flattened_weights = weights.flatten()
        permuted_indices = torch.randperm(flattened_weights.size(0))
        shuffled_weights = flattened_weights[permuted_indices].view(weight_shape)
        return shuffled_weights
    with torch.no_grad():
        for param in model.parameters():
            if param.requires_grad:
                param.data = shuffle_weights(param.data)
    return model

