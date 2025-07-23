from typing import Callable, Optional, Sequence
import torch
import torch.nn as nn
import torch.nn.functional as F

def default_init():
    return nn.init.xavier_uniform_

def get_weight_decay_mask(model):
    """Create weight decay mask for PyTorch model parameters"""
    decay_params = []
    no_decay_params = []
    
    for name, param in model.named_parameters():
        if 'bias' in name or 'Input' in name or 'Output' in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    
    return decay_params, no_decay_params

class MLP(nn.Module):
    def __init__(self,
                 hidden_dims: Sequence[int],
                 activations: Callable = F.relu,
                 activate_final: bool = False,
                 use_layer_norm: bool = False,
                 scale_final: Optional[float] = None,
                 dropout_rate: Optional[float] = None):
        super().__init__()
        self.hidden_dims = hidden_dims
        self.activations = activations
        self.activate_final = activate_final
        self.use_layer_norm = use_layer_norm
        self.scale_final = scale_final
        self.dropout_rate = dropout_rate
        
        layers = []
        
        if self.use_layer_norm and len(hidden_dims) > 0:
            self.input_layer_norm = nn.LayerNorm(hidden_dims[0])
        
        for i in range(len(hidden_dims) - 1):
            in_features = hidden_dims[i]
            out_features = hidden_dims[i + 1]
            
            if i == len(hidden_dims) - 2 and self.scale_final is not None:
                layer = nn.Linear(in_features, out_features)
                nn.init.xavier_uniform_(layer.weight, gain=self.scale_final)
            else:
                layer = nn.Linear(in_features, out_features)
                nn.init.xavier_uniform_(layer.weight)
            
            nn.init.zeros_(layer.bias)
            layers.append(layer)
            
            # Add dropout and activation if not the final layer, or if activate_final is True
            if i < len(hidden_dims) - 2 or self.activate_final:
                if self.dropout_rate is not None and self.dropout_rate > 0:
                    layers.append(nn.Dropout(self.dropout_rate))
        
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor, training: bool = False) -> torch.Tensor:
        if hasattr(self, 'input_layer_norm'):
            x = self.input_layer_norm(x)
            
        layer_idx = 0
        for i in range(len(self.hidden_dims) - 1):
            x = self.layers[layer_idx](x)
            layer_idx += 1
            
            # Apply dropout and activation if not the final layer, or if activate_final is True
            if i < len(self.hidden_dims) - 2 or self.activate_final:
                if self.dropout_rate is not None and self.dropout_rate > 0:
                    if training and layer_idx < len(self.layers):
                        x = self.layers[layer_idx](x)
                        layer_idx += 1
                x = self.activations(x)
        
        return x