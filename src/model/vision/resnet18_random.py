import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F


class ResNet18Encoder(nn.Module):
    def __init__(
        self,
        pretrained: bool = True,
        output_dim: int = 512,
        unit_norm: bool = False,
    ):
        super().__init__()
        resnet = models.resnet18(pretrained=pretrained)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])  # (B, 512, 1, 1)
        self.flatten = nn.Flatten()
        self.unit_norm = unit_norm
        self.output_dim = output_dim
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        
    def forward(self, x):
        """
        x: Tensor of shape (B, 3, H, W) or (..., 3, H, W)
        """
        original_shape = x.shape
        if x.dim() > 4:
            x = x.reshape(-1, *original_shape[-3:])

        # normalize to Imagenet range
        x = (x - self.mean.to(x.device)) / self.std.to(x.device)

        out = self.feature_extractor(x)  # (B, 512, 1, 1)
        out = self.flatten(out)  # (B, 512)

        if self.unit_norm:
            out = F.normalize(out, p=2, dim=-1)

        if len(original_shape) > 4:
            out = out.reshape(*original_shape[:-3], -1)
        return out
