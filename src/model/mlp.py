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

        self.layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()

        for i in range(len(hidden_dims) - 1):
            in_features = hidden_dims[i]
            out_features = hidden_dims[i + 1]

            layer = nn.Linear(in_features, out_features)
            if i == len(hidden_dims) - 2 and self.scale_final is not None:
                nn.init.xavier_uniform_(layer.weight, gain=self.scale_final)
            else:
                nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
            self.layers.append(layer)

            # 각 레이어마다 LayerNorm을 옵션으로 추가
            if self.use_layer_norm:
                self.layer_norms.append(nn.LayerNorm(out_features))
            else:
                self.layer_norms.append(nn.Identity())

        # Dropout은 forward에서만 사용
        self.dropout = nn.Dropout(self.dropout_rate) if self.dropout_rate is not None else nn.Identity()

    def forward(self, x: torch.Tensor, training: bool = False) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            x = layer(x)

            # Activation 적용
            if i < len(self.layers) - 1 or self.activate_final:
                # Dropout 적용
                if self.dropout_rate is not None and self.dropout_rate > 0:
                    if training:
                        # print("dropOut")
                        x = self.dropout(x)

                # LayerNorm 적용
                if self.use_layer_norm:
                    # print("layerNorm")
                    x = self.layer_norms[i](x)
                x = self.activations(x)

        return x