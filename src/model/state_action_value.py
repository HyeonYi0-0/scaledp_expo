import torch
import torch.nn as nn

def default_init():
    return nn.init.xavier_uniform_

class StateActionValue(nn.Module):
    def __init__(self, base_cls, input_dim, **kwargs):
        super().__init__()
        # Extract hidden_dims from kwargs to avoid duplication
        hidden_dims = kwargs.pop('hidden_dims', [256, 256, 256])
        
        # Create the base network with proper input dimension
        self.base_network = base_cls(hidden_dims=[input_dim] + list(hidden_dims), **kwargs)
        
        # Value head - maps from last hidden dim to 1
        last_hidden_dim = hidden_dims[-1]
        self.value_head = nn.Linear(last_hidden_dim, 1)
        nn.init.xavier_uniform_(self.value_head.weight)
        nn.init.zeros_(self.value_head.bias)

    def forward(self, observations: torch.Tensor, actions: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        inputs = torch.cat([observations, actions], dim=-1)
        outputs = self.base_network(inputs, *args, **kwargs)
        value = self.value_head(outputs)
        return value.squeeze(-1)