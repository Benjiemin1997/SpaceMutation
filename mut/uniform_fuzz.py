import torch

def uniform_fuzz_weight(model, lower_bound=-0.1, upper_bound=0.1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    def perturb_model_weights_uniform(weights, lower_bound=-0.1, upper_bound=0.1):
        noise = torch.rand(weights.size()).to(weights) * (upper_bound - lower_bound) + lower_bound
        return weights + noise
    with torch.no_grad():
        for param in model.parameters():
            if param.requires_grad:
                param.data = perturb_model_weights_uniform(param.data, lower_bound, upper_bound)
    return model



