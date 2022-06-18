from typing import Tuple
import torch
from torch import nn

class MlpNet(nn.Module):
    def __init__(self, output_dimension) -> None:
        super(MlpNet, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(output_dimension * 4, output_dimension * 2),
            nn.ReLU(),
            nn.Linear(output_dimension * 2, output_dimension * 2),
            nn.ReLU(),
            nn.Linear(output_dimension * 2, output_dimension),
        )

    def forward(self, input):
        x = self.flatten(input)
        return self.linear_relu_stack(input)


class CgNet(nn.Module):
    def __init__(self, internal_embed_size: int,
                 element_size: int) -> None:
        super(CgNet, self).__init__()
        self.ctx_mlp = MlpNet(internal_embed_size)
        self.elem_mlp = MlpNet(internal_embed_size)
        self.pooling = nn.MaxPool1d(element_size)
        self.embed_size = internal_embed_size

    def forward(
        self, elements: torch.Tensor,
        context: torch.Tensor) -> Tuple[torch.Tensor]:
        """Forward function for one context gating component.
        
        Args:
            elements: tensor whose dimensions are [B, num_elem, ...]
            context: tensor whose dimensions are [B, ...]

        Returns:
            Tuple of two tensors. The first one is the updated elments,
            whose dimensions are [B, num_elem, embed_size]. The second
            is the updated context whose dimensions are [B, embed_size].
        """
        batch_size = elements.shape[0]
        num_elem = elements.shape[1]
        elements = elements.view(batch_size * num_elem, -1)
        elements_mlps = self.elem_mlp(elements)
        elements_mlps = elements_mlps.view(batch_size, num_elem, -1)
        context_mlp = self.ctx_mlp(context)
        elem_internal_dim = elements_mlps.shape[2]
        ctx_internal_dim = context_mlp.shape[1]
        assert elem_internal_dim == ctx_internal_dim
        elements_mul = torch.mul(elements_mlps, context_mlp)
        ctx_updated = self.pooling(elements_mul.permute(0, 2, 1))
        ctx_updated = ctx_updated.view(batch_size, self.embed_size)
        return elements_mul, ctx_updated

