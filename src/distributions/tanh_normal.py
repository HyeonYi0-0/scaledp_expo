import functools
from typing import Optional, Type, Any

import torch
import torch.nn as nn
import torch.distributions as dist
from torch.distributions import Normal as TorchNormal, Independent, TransformedDistribution
from torch.distributions.transforms import TanhTransform
from torch.nn import functional as F

def default_init():
    return nn.init.xavier_uniform_

class TanhTransformedDistribution(TransformedDistribution):
    def __init__(self, base_distribution: dist.Distribution):
        super().__init__(base_distribution, TanhTransform())

    def mode(self) -> torch.Tensor:
        # For Independent(Normal), access the mean through base_dist.base_dist.loc
        if hasattr(self.base_dist, 'base_dist'):
            # Independent distribution case
            mean = self.base_dist.base_dist.loc
        elif hasattr(self.base_dist, 'loc'):
            # Direct Normal distribution case
            mean = self.base_dist.loc
        else:
            # Fallback: try to get mean/mode
            mean = getattr(self.base_dist, 'mean', getattr(self.base_dist, 'mode', None))
        
        return torch.tanh(mean)


class Normal(nn.Module):
    def __init__(self, 
                 base_cls: Type[nn.Module],
                 action_dim: int,
                 log_std_min: Optional[float] = -2,
                 log_std_max: Optional[float] = 2,
                 state_dependent_std: bool = True,
                 squash_tanh: bool = False):
        super().__init__()
        self.base_cls = base_cls
        self.action_dim = action_dim
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.state_dependent_std = state_dependent_std
        self.squash_tanh = squash_tanh
        
        # Create the base network
        self.backbone = base_cls()
        
        # Output layers - get output size from backbone
        # Try to infer the output size from the backbone
        with torch.no_grad():
            dummy_input = torch.randn(1, 10)  # dummy input to infer size
            dummy_input2 = torch.randn(1, 10)  # another dummy input for robustness
            try:
                dummy_output = self.backbone(dummy_input, dummy_input2)
                backbone_output_size = dummy_output.shape[-1]
            except:
                # Fallback to hidden_dims if available
                if hasattr(self.backbone, 'hidden_dims') and len(self.backbone.hidden_dims) > 0:
                    backbone_output_size = self.backbone.hidden_dims[-1]
                else:
                    backbone_output_size = 256  # Default fallback
            
        self.mean_head = nn.Linear(backbone_output_size, action_dim)
        
        if self.state_dependent_std:
            self.log_std_head = nn.Linear(backbone_output_size, action_dim)
        else:
            self.log_std = nn.Parameter(torch.zeros(action_dim))
        
        # Initialize weights
        nn.init.xavier_uniform_(self.mean_head.weight)
        nn.init.zeros_(self.mean_head.bias)
        
        if self.state_dependent_std:
            nn.init.xavier_uniform_(self.log_std_head.weight)
            nn.init.zeros_(self.log_std_head.bias)

    def forward(self, observation, action, *args, **kwargs) -> dist.Distribution:
        x = self.backbone(observation, action, *args, **kwargs)

        means = self.mean_head(x)
        
        if self.state_dependent_std:
            log_stds = self.log_std_head(x)
        else:
            log_stds = self.log_std.expand_as(means)

        log_stds = torch.clamp(log_stds, self.log_std_min, self.log_std_max)

        std = F.softplus(log_stds) + 1e-6
        distribution = Independent(TorchNormal(loc=means, scale=std), 1)

        if self.squash_tanh:
            return TanhTransformedDistribution(distribution)
        else:
            return distribution

    def get_mode(self, inputs, *args, **kwargs) -> torch.Tensor:
        """Get the mode of the distribution"""
        x = self.backbone(inputs, *args, **kwargs)
        means = self.mean_head(x)
        
        if self.squash_tanh:
            return torch.tanh(means)
        else:
            return means


def TanhNormal(base_cls, action_dim: int, **kwargs):
    """Factory function to create TanhNormal policy"""
    return Normal(base_cls, action_dim, squash_tanh=True, **kwargs)