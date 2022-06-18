from typing import Tuple
import torch
from torch import nn


# MLP layer.
class MlpNet(nn.Module):
    def __init__(self, input_dimension: int, 
                 output_dimension: int) -> None:
        super(MlpNet, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_dimension, output_dimension * 4),
            nn.ReLU(),
            nn.Linear(output_dimension * 4, output_dimension * 2),
            nn.ReLU(),
            nn.Linear(output_dimension * 2, output_dimension),
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = self.flatten(input)
        return self.linear_relu_stack(input)


# Context gating layer, which is based on multipath++ paper.
class CgNet(nn.Module):
    def __init__(self, element_attribution_dim: int,
                 context_attribution_dim: int,
                 internal_embed_size: int) -> None:
        super(CgNet, self).__init__()
        self.elem_mlp = MlpNet(
            input_dimension=element_attribution_dim,
            output_dimension=internal_embed_size)
        self.ctx_mlp = MlpNet(
            input_dimension=context_attribution_dim,
            output_dimension=internal_embed_size)
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
        elements = elements.reshape(batch_size * num_elem, -1)
        elements_mlps = self.elem_mlp(elements)
        elements_mlps = elements_mlps.view(batch_size, num_elem, -1)
        context_mlp = self.ctx_mlp(context)
        elem_internal_dim = elements_mlps.shape[2]
        ctx_internal_dim = context_mlp.shape[1]
        assert elem_internal_dim == ctx_internal_dim
        assert self.embed_size == ctx_internal_dim
        elements_mul = torch.mul(elements_mlps,
                                 context_mlp.view(batch_size, 1, 
                                                  ctx_internal_dim))
        # Performs max pooling along the 2nd dimension.
        ctx_updated, _ = torch.max(elements_mul, dim = 1)
        return elements_mul, ctx_updated


# Multi-context gating layer, which is based on multipath++ paper.
class McgNet(nn.Module):
    def __init__(self, element_attribution_dim: int,
                 context_attribution_dim: int,
                 internal_embed_size: int, num_cg: int) -> None:
        super(McgNet, self).__init__()
        self.cg_list = [CgNet(element_attribution_dim,
                              context_attribution_dim,
                              internal_embed_size)]
        if (num_cg > 1):
            self.cg_list.extend(
                [CgNet(internal_embed_size, internal_embed_size, 
                       internal_embed_size) for i in range(1, num_cg)])
        self.num_cg = num_cg

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
        updated_elements, updated_context = self.cg_list[0](elements, context)
        for i in range(1, self.num_cg):
            updated_elements, updated_context = self.cg_list[i](
                updated_elements, updated_context)

        return updated_elements, updated_context


# Learnable query decoder which is based on multipath++ paper.
class LearnableQuery(nn.Module):
    def __init__(self, num_query: int, query_dim: int,
                 context_dim: int, internal_embed_size) -> None:
        super(LearnableQuery, self).__init__()
        self.query = torch.nn.Parameter(torch.randn(num_query, query_dim))
        self.mcg = McgNet(query_dim, context_dim,
                          internal_embed_size=internal_embed_size,
                          num_cg=3)
        self.query_dim = query_dim
        self.num_query = num_query

    def forward(self, context: torch.Tensor) -> Tuple[torch.Tensor]:
        """Forward function for one learnable query.
        
        Args:
            context: tensor whose dimensions are [B, ...]

        Returns:
            Two decoded tensors. The 1st one is the trajectories, whose
            dimensions are [B, num_query, internal_embed_size]. The 2nd
            one is the context, whose dimensions are [B, internal_embed_size].
        """
        batch_size = context.shape[0]
        expanded_query = self.query.view(
            1, self.num_query, self.query_dim)
        expanded_query = expanded_query.expand(
            batch_size, self.num_query, self.query_dim)
        return self.mcg(expanded_query, context)
