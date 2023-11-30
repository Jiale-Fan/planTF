import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import TransformerDecoderLayer
import numpy as np

def generate_tgt_masks(ori_mask, num_modes, num_heads):
    B = ori_mask.shape[0]
    tgt_mask = ori_mask.unsqueeze(-1) & ori_mask.unsqueeze(-2)
    tgt_mask = tgt_mask.unsqueeze(1).repeat(1,num_modes*num_heads,1,1).view(B*num_modes*num_heads, -1, tgt_mask.shape[-1])
    return tgt_mask

def generate_memory_masks(agent_mask, map_mask, num_modes, num_heads):
    B = agent_mask.shape[0]
    memory_mask = agent_mask.unsqueeze(-1) & map_mask.unsqueeze(-2)
    memory_mask = memory_mask.unsqueeze(1).repeat(1,num_heads*num_modes,1,1).view(B*num_heads*num_modes, -1, memory_mask.shape[-1])
    return memory_mask

class TrajectoryDecoder(nn.Module):
    def __init__(self, embed_dim, num_modes, future_steps, out_channels, num_heads=8, dropout=0.1) -> None:
        super().__init__()

        self.embed_dim = embed_dim
        self.num_modes = num_modes
        self.future_steps = future_steps
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.dropout = dropout

        init_ = lambda m: init(m, nn.init.xavier_normal_, lambda x: nn.init.constant_(x, 0), np.sqrt(2))

        self.learned_query = nn.Parameter(torch.Tensor(num_modes, embed_dim), requires_grad=True) # TODO: check if xavier init is possible
        self.learned_query_prob = nn.Parameter(torch.Tensor(num_modes, embed_dim), requires_grad=True)

        nn.init.xavier_normal_(self.learned_query)
        nn.init.xavier_normal_(self.learned_query_prob)

        self.transformer_decoder = TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=self.num_heads,
            dim_feedforward=2048,
            dropout=self.dropout,
            activation="relu",
            batch_first=True,
        )

        self.output_model = OutputModel(d_k=self.embed_dim, predict_yaw=True, future_steps=self.future_steps)

        self.mode_map_attn = nn.MultiheadAttention(self.embed_dim, num_heads=self.num_heads, dropout=self.dropout, batch_first=True)
        self.prob_decoder = nn.MultiheadAttention(self.embed_dim, num_heads=self.num_heads, dropout=self.dropout, batch_first=True)

        self.prob_predictor = init_(nn.Linear(self.embed_dim, 1))

    def forward(self, agent_emb, map_emb, agent_mask, map_mask):

        # assert not torch.isnan(agent_emb).any()
        # assert not torch.isnan(map_emb).any()

        B, A = agent_emb.shape[:2]
        modal_specific_agent_emb = self.learned_query[None,:,None,:]+agent_emb[:,None,:,:] # [B, num_modes, ego+agents, embed_dim]
        modal_specific_agent_emb = modal_specific_agent_emb.view(-1, A, self.embed_dim)

        memory_map_emb = map_emb.unsqueeze(1).repeat(1, self.num_modes, 1, 1).view(-1, map_emb.shape[-2], map_emb.shape[-1])

        x = self.transformer_decoder(tgt=modal_specific_agent_emb, memory=memory_map_emb,
                                      tgt_mask=generate_tgt_masks(agent_mask, self.num_modes, self.num_heads), 
                                      memory_mask=generate_memory_masks(agent_mask, map_mask, self.num_modes, self.num_heads))
        x = x.view(B, self.num_modes, -1, self.embed_dim) # [B, num_modes, ego+agents, embed_dim]

        predictions = self.output_model(x) # [B, num_modes, ego+agents, 6]

        P = self.learned_query_prob.unsqueeze(0).repeat(B, 1, 1)

        mode_params_emb = self.prob_decoder(query=P, key=agent_emb,
                                            value=agent_emb, key_padding_mask=agent_mask)[0] # TODO: add valid masks
        mode_params_emb = self.mode_map_attn(query=mode_params_emb, key=map_emb, value=map_emb,
                                                key_padding_mask=map_mask
                                                )[0] + mode_params_emb
        mode_probs = self.prob_predictor(mode_params_emb).squeeze(-1)
        mode_probs = F.softmax(mode_probs, dim=-1)

        # assert not torch.isnan(predictions).any()
        # assert not torch.isnan(mode_probs).any()

        return predictions, mode_probs


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module

class OutputModel(nn.Module):
    '''
    This class operates on the output of AutoBot-Joint's decoder representation. It produces the parameters of a
    bivariate Gaussian distribution and possibly predicts the yaw.
    '''
    def __init__(self, d_k=64, predict_yaw=False, future_steps=80):
        super(OutputModel, self).__init__()
        self.d_k = d_k
        self.predict_yaw = predict_yaw
        self.future_steps = future_steps
        out_len = 5
        if predict_yaw:
            out_len = 6

        init_ = lambda m: init(m, nn.init.xavier_normal_, lambda x: nn.init.constant_(x, 0), np.sqrt(2))

        self.hidden_dim = 2 * self.d_k

        self.observation_model = nn.Sequential(
            init_(nn.Linear(self.d_k, self.hidden_dim)), nn.ReLU(),
            init_(nn.Linear(self.hidden_dim, self.hidden_dim)), nn.ReLU(),
            init_(nn.Linear(self.hidden_dim, future_steps*out_len))
        )
        self.min_stdev = 0.01

    def forward(self, agent_latent_state):
        '''
        :param agent_latent_state: [B, num_modes, ego+agents, embed_dim]
        '''
        B, modes, agents, _ = agent_latent_state.shape
        pred_obs = self.observation_model(agent_latent_state).reshape(B, modes, agents, self.future_steps, -1)
        x_mean = pred_obs[..., 0]
        y_mean = pred_obs[..., 1]
        x_sigma = F.softplus(pred_obs[..., 2]) + self.min_stdev
        y_sigma = F.softplus(pred_obs[..., 3]) + self.min_stdev
        rho = torch.tanh(pred_obs[..., 4]) * 0.9  # for stability
        if self.predict_yaw:
            # yaws = torch.clip(pred_obs[..., 5], -np.pi, np.pi) 
            yaws = pred_obs[..., 5] # TODO: check if this needs clipping
            return torch.stack([x_mean, y_mean, x_sigma, y_sigma, rho, yaws], dim=-1)
        else:
            return torch.stack([x_mean, y_mean, x_sigma, y_sigma, rho], dim=-1)