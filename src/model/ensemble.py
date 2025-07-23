from typing import Type
import torch
import torch.nn as nn

class Ensemble(nn.Module):
    def __init__(self, net_cls: Type[nn.Module], num: int = 2, **kwargs):
        super().__init__()
        self.net_cls = net_cls
        self.num = num
        
        # Create ensemble of networks
        self.networks = nn.ModuleList([
            net_cls(**kwargs) for _ in range(num)
        ])

    def forward(self, *args, **kwargs):
        outputs = []
        for network in self.networks:
            output = network(*args, **kwargs)
            outputs.append(output)
        return torch.stack(outputs, dim=0)

def subsample_ensemble(networks, num_sample: int, num_qs: int, device='cpu'):
    """Subsample ensemble networks for REDQ-style training"""
    if num_sample is not None and num_sample < num_qs:
        indices = torch.randperm(num_qs, device=device)[:num_sample]
        # Return a subset of networks based on indices
        return [networks[i] for i in indices]
    return networks