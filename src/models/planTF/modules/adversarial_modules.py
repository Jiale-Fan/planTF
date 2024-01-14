import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..layers.transformer_encoder_layer import TransformerEncoderLayer
from ..layers.common_layers import build_mlp


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