from torch import nn
import torch

def get_model_unstructured_sparsity(model):
    total = []
    pruned = []

    for module in model.modules():
        if hasattr(module, "weight") and isinstance(module, nn.Conv2d):
            total.append(module.weight.nelement())
            pruned.append(torch.sum(module.weight == 0))

    total = float(sum(total))
    pruned = float(sum(pruned))

    sparsity = 100 * (pruned / total)

    return total, pruned, sparsity
