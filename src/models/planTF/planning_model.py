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
from .modules.agent_encoder import AgentEncoder, AgentInteractionEncoder
from .modules.map_encoder import MapEncoder
from .modules.trajectory_decoder import TrajectoryDecoder

import numpy as np
from analysis.visualization import plot_sample_elements
import os
from .layers.embedding import NATSequenceEncoder

# no meaning, required by nuplan
trajectory_sampling = TrajectorySampling(num_poses=8, time_horizon=8, interval_length=1)

init_ = lambda m: init(m, nn.init.xavier_normal_, lambda x: nn.init.constant_(x, 0), np.sqrt(2))

def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module

class ProjHead(nn.Module):
    '''
    Nonlinear projection head that maps the extracted motion features to the embedding space
    '''
    def __init__(self, feat_dim, hidden_dim, head_dim):
        super(ProjHead, self).__init__()
        self.head = nn.Sequential(
            init_(nn.Linear(feat_dim, hidden_dim)),
            nn.ReLU(inplace=True),
            init_(nn.Linear(hidden_dim, head_dim))
            )

    def forward(self, feat):
        return self.head(feat)


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
        # state_dropout=0.75, # not in effect now
        feature_builder: NuplanFeatureBuilder = NuplanFeatureBuilder(),
        projection_dim = 256,
        plot=False,
    ) -> None:
        super().__init__(
            feature_builders=[feature_builder],
            target_builders=[EgoTrajectoryTargetBuilder(trajectory_sampling)],
            future_trajectory_sampling=trajectory_sampling,
        )

        self.dim = dim
        self.history_steps = history_steps
        self.future_steps = future_steps
        self.num_modes = num_modes
        self.plot = plot
        self.sample_index = 0

        self.pos_emb = build_mlp(4, [dim] * 2)
        self.pos_emb_route = build_mlp(4, [dim] * 2)

        self.agent_encoder_pred = AgentEncoder(
            state_channel=6,
            history_channel=history_channel,
            dim=dim,
            hist_steps=history_steps,
            drop_path=drop_path,
            use_ego_history=True,
            state_attn_encoder=state_attn_encoder,
            state_dropout=0,
        )

        self.agent_encoder_plan = AgentEncoder(
            state_channel=3,
            history_channel=history_channel,
            dim=dim,
            hist_steps=history_steps,
            drop_path=drop_path,
            use_ego_history=False,
            state_attn_encoder=state_attn_encoder,
            state_dropout=0,
        )

        self.map_encoder = MapEncoder(
            dim=dim,
            polygon_channel=polygon_channel,
        )

        self.encoder_blocks = nn.ModuleList(
            TransformerEncoderLayer(dim=dim, num_heads=num_heads, drop_path=dp)
            for dp in [x.item() for x in torch.linspace(0, drop_path, encoder_depth)]
        )
        self.norm = nn.LayerNorm(dim)

        self.trajectory_decoder_pred = TrajectoryDecoder(
            embed_dim=dim,
            num_modes=num_modes,
            future_steps=future_steps,
            out_channels=4,
        )

        self.trajectory_decoder_plan = TrajectoryDecoder(
            embed_dim=dim,
            num_modes=1,
            future_steps=future_steps,
            out_channels=4,
        )

        self.interaction_encoder = AgentInteractionEncoder(
            future_channel=6,
            future_steps=future_steps,
            dim=dim,
            drop_path=drop_path,
        )

        self.mode_fusion_attention = nn.MultiheadAttention(dim, num_heads=num_heads, dropout=0.1, batch_first=True)

        self.learned_query = nn.Parameter(torch.Tensor(num_modes, dim).to('cuda'), requires_grad=True) # TODO: check if xavier init is possible
        nn.init.xavier_normal_(self.learned_query)

        # self.agent_predictor = build_mlp(dim, [dim * 2, future_steps * 2], norm="ln")

        self.scenario_embedding = nn.Parameter(torch.randn(1, 5, dim))
        nn.init.xavier_normal_(self.scenario_embedding)

        # self.scenario_projector = ProjHead(feat_dim=dim, hidden_dim=dim//4, head_dim=8)
        self.env_projector = ProjHead(feat_dim=dim, hidden_dim=dim, head_dim=projection_dim)
        self.beh_projector = ProjHead(feat_dim=dim, hidden_dim=dim, head_dim=projection_dim)
        self.obj_projector = ProjHead(feat_dim=dim, hidden_dim=dim, head_dim=projection_dim)

        self.scene_target_projector = ProjHead(feat_dim=dim*2, hidden_dim=dim, head_dim=projection_dim)
        self.target_encoder = init_(nn.Linear(future_steps*3, dim))

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

    def forward(self, data):
        agent_pos = data["agent"]["position"][:, :, self.history_steps - 1]
        agent_heading = data["agent"]["heading"][:, :, self.history_steps - 1]
        agent_mask = data["agent"]["valid_mask"][:, :, : self.history_steps]
        polygon_center = data["map"]["polygon_center"]
        polygon_mask = data["map"]["valid_mask"]
        agent_mask = data["agent"]["valid_mask"][:, :, :self.history_steps]

        bs, A = agent_pos.shape[0:2]

        position = torch.cat([agent_pos, polygon_center[..., :2]], dim=1)
        angle = torch.cat([agent_heading, polygon_center[..., 2]], dim=1)
        pos = torch.cat(
            [position, torch.stack([angle.cos(), angle.sin()], dim=-1)], dim=-1
        )
        pos_embed = self.pos_emb(pos)

        agent_key_padding = ~(agent_mask.any(-1))
        polygon_key_padding = ~(polygon_mask.any(-1))

        # add fake key_padding_mask for scenario embedding TODO: check if mask should be inverse mask
        scenario_emb_key_padding = torch.zeros(bs, self.scenario_embedding.shape[1], dtype=torch.bool, device=agent_key_padding.device)

        key_padding_mask = torch.cat([agent_key_padding, polygon_key_padding, scenario_emb_key_padding], dim=-1)

        x_agent_pred = self.agent_encoder_pred(data)
        x_polygon_all= self.map_encoder(data, on_route_embedding=False)

        # add learnable initial scenario embedding
        scenario_emb = self.scenario_embedding.repeat(bs, 1, 1)

        ##### 1. prediction forward pass #####
        
        x_pred = torch.cat([x_agent_pred, x_polygon_all], dim=1) + pos_embed 
        x_pred = torch.cat([x_pred, scenario_emb], dim=1)
        # x: [batch, n_elem, n_dim]. n_elem is not a fixed number, it depends on the number of agents and polygons in the scene

        for blk in self.encoder_blocks:
            x_pred = blk(x_pred, key_padding_mask=key_padding_mask)
        x_pred = self.norm(x_pred)

        # predictions, probabilities = self.trajectory_decoder(x[:, 0:A], x[:, A:-4], agent_key_padding, polygon_key_padding) # x: [batch, n_elem, 128], trajectory: [batch, modal, 80, 4], probability: [batch, 6]
        polygon_sceemb_key_padding = torch.cat([polygon_key_padding, scenario_emb_key_padding], dim=-1)
        predictions, context, _ = self.trajectory_decoder_pred(x_pred[:, 0:A], x_pred[:, A:], agent_key_padding, polygon_sceemb_key_padding, self.learned_query)
        # prediction = self.agent_predictor(x[:, 1:A]).view(bs, -1, self.future_steps, 2)

        interaction_embedding = self.interaction_encoder(predictions, agent_mask)
        ##### 2. planning forward pass #####


        x_agent_plan = self.agent_encoder_plan(data)
        x_polygon_plan= self.map_encoder(data, on_route_embedding=True)

        x_plan = torch.cat([x_agent_plan, x_polygon_plan], dim=1) + pos_embed
        x_plan = torch.cat([x_plan, scenario_emb], dim=1)
        # x: [batch, n_elem, n_dim]. n_elem is not a fixed number, it depends on the number of agents and polygons in the scene

        for blk in self.encoder_blocks:
            x_plan = blk(x_plan, key_padding_mask=key_padding_mask)
        x_plan = self.norm(x_plan)

        mode_selection_query = x_plan[:, -5]
        plan_query, _ = self.mode_fusion_attention(
            query = mode_selection_query.unsqueeze(1),
            key = interaction_embedding,
            value = interaction_embedding,
        )

        plans, _, probabilities = self.trajectory_decoder_plan(x_plan[:, 0:1], x_plan[:, A:], agent_key_padding[:, 0:1], polygon_sceemb_key_padding, plan_query)
        plans = plans.squeeze(2)

        # get the projection of the scenario embedding
        scenario_emb = x_plan[:,-1] # [B, dim]

        # get the projection of the scenario feature embeddings

        # beh_proj = self.beh_projector(x_plan[:, -2])
        # env_proj = self.env_projector(x_plan[:, -3])
        # obj_proj = self.obj_projector(x_plan[:, -4])


        out = {
            "trajectory": plans,
            "probability": probabilities,
            "prediction": predictions,
            # "beh_proj": beh_proj,
            # "env_proj": env_proj,
            # "obj_proj": obj_proj,
        }

        

        # if self.training:
        #     trajectory = plans[:, :, 0, :, torch.Tensor([0,1,5]).to(torch.long)] # [B, num_modes, timestep, states_dim]
        #     trajectory_embs = self.target_encoder(trajectory.view(bs, self.num_modes, -1)) # [B, num_modes, 128]
        #     trajectory_embs = trajectory_embs.view((-1, self.dim)).unsqueeze(0).repeat(bs, 1, 1) # [B, B*num_modes, 128]

        #     scene_target_emb = torch.cat([trajectory_embs, scenario_emb.unsqueeze(1).repeat(1,bs*self.num_modes,1)], dim=-1) # [B, B*num_modes, 256]
        #     scene_target_emb_projs = self.scene_target_projector(scene_target_emb) # [B, B*num_modes, 8]
        #     out["scene_plan_emb_proj"] = scene_target_emb_projs# [B, B*num_modes, 8]

        #     target_emb = self.target_encoder(data["agent"]["target"][:,0].reshape(bs, -1))
        #     scene_target_emb = torch.cat([target_emb, scenario_emb], dim=-1)
        #     scene_target_emb_proj = self.scene_target_projector(scene_target_emb)
        #     out["scene_target_emb_proj"] = scene_target_emb_proj # [B, 8]

        #     ego_target = data["agent"]["target"][:, 0]
        #     errors = (ego_target[:, None, :, :2] - plans[:, :, 0, :, :2]).norm(dim=-1).sum(dim=(-1))
        #     closest_mode = errors.argmin(dim=-1)

        #     output_trajectory = plans[:, :, 0][torch.arange(bs), closest_mode]
        #     best_emb = self.target_encoder(output_trajectory[..., torch.Tensor([0,1,5]).to(torch.long)].reshape(bs, -1))
        #     scene_best_emb = torch.cat([best_emb, scenario_emb], dim=-1)
        #     scene_best_emb_proj = self.scene_target_projector(scene_best_emb)
        #     out["scene_best_emb_proj"] = scene_best_emb_proj


        if not self.training:
      
            output_trajectory = plans[:, 0]
            output_trajectory = output_trajectory[..., torch.Tensor([0,1,5]).to(torch.long)] # [B, timestep, states_dim]
            out["output_trajectory"] = output_trajectory

            if self.plot:
                # examine whether the path exists
                save_path = './debug_files/scenario_2/'
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                np_data = NuplanFeature(data=data).to_numpy().data
                plot_sample_elements(np_data, out, sample_index=self.sample_index, save_path=save_path)
                self.sample_index += 1
                print('plotting: ', self.sample_index)

        return out
