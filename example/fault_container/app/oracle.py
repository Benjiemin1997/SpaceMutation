import torch
from torch import nn

from mut.guassian_fuzz_splayers import gaussian_fuzzing_splayer


def get_first_conv_weight(model):
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            return module.weight.detach().clone()
    raise RuntimeError("No Conv2d layer found in model")


def gaussian_oracle(model):
    w0 = get_first_conv_weight(model)

    mutated_model = gaussian_fuzzing_splayer(
        model,
        std_ratio=0.8,
        target_layer_type=nn.Conv2d
    )

    if mutated_model is None:
        raise RuntimeError("Mutation operator returned None")

    w1 = get_first_conv_weight(mutated_model)

    if torch.equal(w0, w1):
        raise AssertionError("Conv2d weights did not change after gaussian fuzzing")

    delta = (w1 - w0).flatten()
    mean_abs = delta.mean().abs().item()

    if mean_abs >= 0.5:
        raise AssertionError(
            f"Noise mean deviates too much from zero: {mean_abs}"
        )

    return {
        "oracle_status": "passed",
        "mean_abs": mean_abs
    }