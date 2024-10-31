import torch

def gaussian_fuzz_weight(model, std_ratio=1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    def perturb_model_weights(weights, std_ratio=1):
        std = torch.std(weights) * std_ratio
        noise = torch.normal(0, std, size=weights.size()).to(weights)
        return weights + noise
    with torch.no_grad():
        for param in model.parameters():
            if param.requires_grad:
                param.data = perturb_model_weights(param.data, std_ratio)
    return model

