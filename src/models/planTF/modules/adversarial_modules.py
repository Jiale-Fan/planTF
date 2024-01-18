import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..layers.transformer_encoder_layer import TransformerEncoderLayer
from ..layers.common_layers import build_mlp


import torch
import torch.nn as nn

class LearnableNoiseModule(nn.Module):
    """
    This module aims to generate stochastic noise from a learnable parameterized Gaussian distribution.
    Reference: arXiv:2308.00566
    """
    def __init__(self, sigma: int=0.25, dim: int=128):
        super(LearnableNoiseModule, self).__init__()
        self.sigma = sigma
        self.A = nn.Linear(dim, dim*dim, bias=False)
        self.b = nn.Parameter(torch.randn(dim))

    def forward(self, context, pos_emb):
        '''
        context: [batch_size, length, dim]
        pos_emb: [batch_size, length, dim]
        '''
        # cov = self.sigma * self.A @ self.A.t()
        noise = torch.randn_like(context) @ self.A
        context_scaled = pos_emb @ self.A + self.b
        out = context_scaled + noise + pos_emb
        return out
    



# class LearnableNoiseModule(nn.Module):
#     """
#     This module aims to generate stochastic noise from a learnable parameterized Gaussian distribution.
#     Reference: arXiv:2308.00566
#     """
#     def __init__(self, sigma: int=0.25, dim: int=128):
#         super(LearnableNoiseModule, self).__init__()
#         self.sigma = sigma
#         self.dim = dim
#         self.A_net = nn.Linear(dim, dim*dim, bias=False)
#         self.b = nn.Parameter(torch.randn(dim))

#     def forward(self, context, pos_emb):
#         '''
#         context: [batch_size, length, dim]
#         pos_emb: [batch_size, length, dim]
#         '''
#         # cov = self.sigma * self.A @ self.A.t()
#         A = self.A_net(pos_emb).view(-1, self.dim*self.dim)
#         noise = torch.matmal(torch.randn_like(context), A)
#         context_scaled = context @ self.A + self.b
#         out = context_scaled + noise + pos_emb
#         return out


class TransformerMasker(nn.Module):
    def __init__(self, in_dim, num_heads=16, mask_rate=0.5, dropout=0.1):
        super(TransformerMasker, self).__init__()
        self.in_dim = in_dim
        self.mask_rate= mask_rate
        
        self.tf = TransformerEncoderLayer(dim=in_dim, num_heads=num_heads, drop_path=dropout)
        self.linear = nn.Linear(in_dim, 1, bias=False)

        # self.bn = nn.BatchNorm1d(num_classes, affine=False)

    def forward(self, f, key_padding_mask):
        '''
            input:
                f: [B, N, in_dim]
                key_padding_mask: [B, N]
            output:
                    z: [B, N]
        '''
        mask_prob = self.linear(self.tf(f, key_padding_mask=key_padding_mask)).squeeze(-1)
        z = torch.zeros_like(mask_prob)
        k = int(mask_prob.shape[1]*self.mask_rate)

        for _ in range(k):
            mask = F.gumbel_softmax(mask_prob, dim=1, tau=0.5, hard=False)
            z = torch.maximum(mask,z)
        return z
    


class AdversarialEmbeddingPerturbator(nn.Module):
    def __init__(self, dim) -> None:
        super(AdversarialEmbeddingPerturbator, self).__init__()
        self.layers = build_mlp(dim, [4*dim, 4*dim, dim])
    def forward(self, x):
        """
        x : [B, N, dim]
        return : [B, N, dim]
        
        """
        y = self.layers(x)
        # clip the 2-norm of the perturbation to be less than the mean 2-norm of the embedding
        mean_embed_norm = torch.mean(torch.norm(x, dim=-1))
        y_ori_scale = torch.norm(y, dim=-1, keepdim=True)
        y_clamped_scale = torch.clamp(y_ori_scale, max=mean_embed_norm)
        y_clamped = y / y_ori_scale * y_clamped_scale
        return y_clamped