# This script implements MABD decoder as described in the paper.

import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttentionDecoder(nn.Module):
    def __init__(self, L_dec=3, d_k=128, num_heads=16, dropout=0.0, tx_hidden_size=384) -> None:
        super().__init__()

        self.L_dec = L_dec
        self.d_k = d_k
        self.num_heads = num_heads
        self.dropout = dropout
        self.tx_hidden_size = tx_hidden_size

        self.Q = nn.Parameter(torch.Tensor(self.T, 1, self.c, 1, self.d_k), requires_grad=True)
        nn.init.xavier_uniform_(self.Q)

        self.social_attn_decoder_layers = []
        self.temporal_attn_decoder_layers = []
        for _ in range(self.L_dec):
            tx_decoder_layer = nn.TransformerDecoderLayer(d_model=self.d_k, nhead=self.num_heads,
                                                          dropout=self.dropout, dim_feedforward=self.tx_hidden_size)
            self.temporal_attn_decoder_layers.append(nn.TransformerDecoder(tx_decoder_layer, num_layers=2))
            tx_encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_k, nhead=self.num_heads,
                                                          dropout=self.dropout, dim_feedforward=self.tx_hidden_size)
            self.social_attn_decoder_layers.append(nn.TransformerEncoder(tx_encoder_layer, num_layers=1))

        self.temporal_attn_decoder_layers = nn.ModuleList(self.temporal_attn_decoder_layers)
        self.social_attn_decoder_layers = nn.ModuleList(self.social_attn_decoder_layers)

    def forward(self, x):
        B = x.shape[0]

        dec_parameters = self.Q.repeat(1, B, 1, self._M+1, 1).view(self.T, B*self.c, self._M+1, -1)
        dec_parameters = torch.cat((dec_parameters, agent_types_features), dim=-1)
        dec_parameters = self.dec_agenttypes_encoder(dec_parameters)
        agents_dec_emb = dec_parameters

        for d in range(self.L_dec):
            if self.use_map_lanes and d == 1:
                agents_dec_emb = agents_dec_emb.reshape(self.T, -1, self.d_k)
                agents_dec_emb_map = self.map_attn_layers(query=agents_dec_emb, key=map_features, value=map_features,
                                                          key_padding_mask=road_segs_masks)[0]
                agents_dec_emb = agents_dec_emb + agents_dec_emb_map
                agents_dec_emb = agents_dec_emb.reshape(self.T, B*self.c, self._M+1, -1)

            agents_dec_emb = self.temporal_attn_decoder_fn(agents_dec_emb, x, opps_masks_modes, layer=self.temporal_attn_decoder_layers[d])
            agents_dec_emb = self.social_attn_decoder_fn(agents_dec_emb, opps_masks_modes, layer=self.social_attn_decoder_layers[d])