import torch
import torch.nn as nn
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from nuplan.planning.training.modeling.torch_module_wrapper import TorchModuleWrapper
from nuplan.planning.training.preprocessing.target_builders.ego_trajectory_target_builder import (
    EgoTrajectoryTargetBuilder,
)

from src.feature_builders.nuplan_feature_builder import NuplanFeatureBuilder
from src.features.nuplan_feature import NuplanFeature

from .layers.common_layers import build_mlp
from .layers.transformer_encoder_layer import TransformerEncoderLayer
from .modules.agent_encoder import AgentEncoder
from .modules.map_encoder import MapEncoder
from .modules.trajectory_decoder import TrajectoryDecoder
import torch.nn.functional as F


# no meaning, required by nuplan
trajectory_sampling = TrajectorySampling(num_poses=8, time_horizon=8, interval_length=1)


class PlanningModel(TorchModuleWrapper):
    def __init__(
        self,
        dim=128,
        state_channel=6,
        polygon_channel=6,
        history_channel=9,
        history_steps=21,
        future_steps=80,
        encoder_depth=4,
        drop_path=0.2,
        num_heads=8,
        num_modes=6,
        use_ego_history=False,
        state_attn_encoder=True,
        state_dropout=0.75,
        ema_update_alpha=0.999,
        mask_rate=0.7,
        alpha_0=0, 
        alpha_T=0.5,
        eval_mode="no_mask", # "best", "random", "worst", or "no_mask"
        feature_builder: NuplanFeatureBuilder = NuplanFeatureBuilder(),
    ) -> None:
        super().__init__(
            feature_builders=[feature_builder],
            target_builders=[EgoTrajectoryTargetBuilder(trajectory_sampling)],
            future_trajectory_sampling=trajectory_sampling,
        )

        self.dim = dim
        self.history_steps = history_steps
        self.future_steps = future_steps
        self.alpha = ema_update_alpha
        self.mask_rate = mask_rate
        self.eval_mode = eval_mode

        self.pos_emb = build_mlp(4, [dim] * 2)
        self.agent_encoder = AgentEncoder(
            state_channel=state_channel,
            history_channel=history_channel,
            dim=dim,
            hist_steps=history_steps,
            drop_path=drop_path,
            use_ego_history=use_ego_history,
            state_attn_encoder=state_attn_encoder,
            state_dropout=state_dropout,
        )

        self.map_encoder = MapEncoder(
            dim=dim,
            polygon_channel=polygon_channel,
        )
        # new stuff

        self.element_type_embedding = nn.Embedding(7, dim)

        self.student_model = TSModel(
            dim=dim,
            future_steps=future_steps,
            encoder_depth=encoder_depth,
            drop_path=drop_path,
            num_heads=num_heads,
            num_modes=num_modes,
        )

        self.teacher_model = TSModel(
            dim=dim,
            future_steps=future_steps,
            encoder_depth=encoder_depth,
            drop_path=drop_path,
            num_heads=num_heads,
            num_modes=num_modes,
        )
        self.teacher_model.requires_grad_(False)

        self.alpha_0 = alpha_0
        self.alpha_T = alpha_T

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def get_mask(self, key_padding_mask, scores=None, fraction=None):
        '''
            input: key_padding_mask: [batch, n_elem]
            output: mask: [batch, n_elem] with mask_rate of its valid elements set to 1
        '''
        if fraction is None:
            fraction = self.mask_rate
        padding_masks = ~key_padding_mask
        valid_num = padding_masks.sum(dim=-1)
        unmasked_num = (valid_num * fraction).to(torch.long)
        if scores is None:
            scores = torch.stack([torch.randperm(padding_masks.shape[-1])+1 for _ in range(padding_masks.shape[0])], dim=0)
        scores = scores.to(padding_masks.device)*padding_masks
        sorted_idx = torch.sort(scores, dim=-1, descending=True)[0]
        indices_split = torch.stack([sorted_idx[i,unmasked_num[i]] for i in range(padding_masks.shape[0])])
        mask = scores > indices_split.unsqueeze(-1)
        return mask.unsqueeze(-1)

    def forward(self, data, progress=None):

        # data preparation
        agent_pos = data["agent"]["position"][:, :, self.history_steps - 1]
        agent_heading = data["agent"]["heading"][:, :, self.history_steps - 1]
        agent_mask = data["agent"]["valid_mask"][:, :, : self.history_steps]
        polygon_center = data["map"]["polygon_center"]
        polygon_mask = data["map"]["valid_mask"]

        bs, A = agent_pos.shape[0:2]

        position = torch.cat([agent_pos, polygon_center[..., :2]], dim=1)
        angle = torch.cat([agent_heading, polygon_center[..., 2]], dim=1)
        pos = torch.cat(
            [position, torch.stack([angle.cos(), angle.sin()], dim=-1)], dim=-1
        )
        pos_embed = self.pos_emb(pos)

        agent_key_padding = ~(agent_mask.any(-1))
        polygon_key_padding = ~(polygon_mask.any(-1))
        key_padding_mask = torch.cat([agent_key_padding, polygon_key_padding], dim=-1)

        x_agent = self.agent_encoder(data)
        x_polygon = self.map_encoder(data)

        x = torch.cat([x_agent, x_polygon], dim=1) + pos_embed

        M = self.get_mask(key_padding_mask)

        if not self.training:
            M = torch.zeros_like(M)
        if progress is not None:
            if progress > 0.5:
                M = torch.zeros_like(M)

        # if not self.training: 
        #     if self.eval_mode == "worst":
        #         _,_,estimation = self.teacher_model(x, M, key_padding_mask)
        #         scores = estimation[..., 0]
        #         M = self.get_mask(key_padding_mask, scores)
        #     elif self.eval_mode == "random":
        #         pass
        #     elif self.eval_mode == "best":
        #         _,_,estimation = self.teacher_model(x, M, key_padding_mask)
        #         scores = estimation[..., 0].pow(-1)
        #         M = self.get_mask(key_padding_mask, scores)
        #     elif self.eval_mode == "no_mask":
        #         M = torch.zeros_like(M)
        #     else:
        #         raise NotImplementedError
        # else:
        #     M = torch.zeros_like(M)
        #     _,_,estimation = self.teacher_model(x, M, key_padding_mask)
        #     scores = estimation[..., 0].pow(-1)
        #     frac = self.alpha_0 + (self.alpha_T - self.alpha_0) * progress
        #     M_score = self.get_mask(key_padding_mask, scores, frac)
        #     padding_mask = key_padding_mask | M_score.squeeze(-1)
        #     M_random = self.get_mask(padding_mask, fraction=self.mask_rate-frac)
        #     M_final = M_score | M_random
        #     M = M_final


        # x_p = (x_initial*(~M)+ (pos_embed+self.masked_embedding_offset)*M)*(~key_padding_mask.unsqueeze(-1)) 
        # masking_rate of the original input are masked

        trajectory, probability, estimation = self.student_model(x, M, key_padding_mask)

        out = {
            "trajectory": trajectory,
            "probability": probability,
            "estimation": estimation,
            "mask": (~M)*(~key_padding_mask.unsqueeze(-1)),
        }


        best_mode = probability.argmax(dim=-1)
        output_trajectory = trajectory[torch.arange(bs), best_mode]
        angle = torch.atan2(output_trajectory[..., 3], output_trajectory[..., 2])
        out["output_trajectory"] = torch.cat(
            [output_trajectory[..., :2], angle.unsqueeze(-1)], dim=-1
        )

        return out
    
    def EMA_update(self):
        for param, param_t in zip(self.student_model.parameters(), self.teacher_model.parameters()):
            param_t.data = self.alpha*param_t.data + (1-self.alpha)*param.data
    
    # def rollout(self, data, output_trajectory):
    #     """
    #     Rollout the planned trajectory, return the ego-centric scene in sparse future frames.
    #         data: dict, input data
    #         planned_trajectory: torch.Tensor, [bs, future_steps, 3]
    #     """
    #     future_keyframes = output_trajectory[:, 0:5:30, :]

    #     data_unpacked = NuplanFeature(data=data).unpack()
    #     bs, frames = future_keyframes.shape[0:2]

    #     for i in range(bs):
    #         normed = []
    #         for j in range(frames):
    #             data_unpacked[i].data["current_state"] = output_trajectory[i, j]
    #             normed.append(data_unpacked[i].normalize())
                


class TSModel(nn.Module):
    def __init__(self, dim=128, future_steps=80, encoder_depth=4, drop_path=0.2, num_heads=8, num_modes=6):
        super().__init__()

        self.encoder_blocks = nn.ModuleList(
            TransformerEncoderLayer(dim=dim, num_heads=num_heads, drop_path=dp)
            for dp in [x.item() for x in torch.linspace(0, drop_path, encoder_depth)]
        )
        self.norm = nn.LayerNorm(dim)

        self.trajectory_decoder = TrajectoryDecoder(
            embed_dim=dim,
            num_modes=num_modes,
            future_steps=future_steps,
            out_channels=4,
        )

        self.estimation_blocks = nn.ModuleList(
            TransformerEncoderLayer(dim=dim, num_heads=num_heads, drop_path=dp)
            for dp in [x.item() for x in torch.linspace(0, drop_path, encoder_depth)]
        )

        # self.masked_embedding_offset = nn.Parameter(torch.randn(1, 1, dim).to('cuda'), requires_grad=True)
        self.ego_embedding = nn.Parameter(torch.randn(1, 1, dim).to('cuda'), requires_grad=True)

        self.estimation_head = build_mlp(dim, [dim * 2, 2], norm="ln")
        self.min_stdev = 0.01

    def forward(self, x_input, mask, key_padding_mask):
        M = mask
        x_e = x_input.clone().detach()
        x = x_input
        # x = (x*(~M)+ (self.masked_embedding_offset)*M)*(~key_padding_mask.unsqueeze(-1)) 

        key_padding_mask = key_padding_mask | M.squeeze(-1)

        x = torch.concat([self.ego_embedding.repeat(x_input.shape[0], 1, 1), x], dim=1)
        key_padding_mask_p = torch.cat([torch.zeros(x_input.shape[0], 1).to('cuda'), key_padding_mask], dim=-1)

        for blk in self.encoder_blocks:
            x = blk(x, key_padding_mask=key_padding_mask_p)
        x = self.norm(x)

        trajectory, probability = self.trajectory_decoder(x[:, 0])

        for blk in self.estimation_blocks:
            x_e = blk(x_e, key_padding_mask=key_padding_mask)
        x_e = self.norm(x_e)
        estimated_error = self.estimation_head(x_e)

        estimated_error_std = F.softplus(estimated_error[..., 1:2]) + self.min_stdev
        estimated_error_ = torch.cat([estimated_error[..., 0:1], estimated_error_std], dim=-1)

        return trajectory, probability, estimated_error_