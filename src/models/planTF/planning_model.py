import torch
import torch.nn as nn
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from nuplan.planning.training.modeling.torch_module_wrapper import TorchModuleWrapper
from nuplan.planning.training.preprocessing.target_builders.ego_trajectory_target_builder import (
    EgoTrajectoryTargetBuilder,
)

from src.feature_builders.nuplan_feature_builder import NuplanFeatureBuilder

from .layers.common_layers import build_mlp
from .layers.transformer_encoder_layer import TransformerEncoderLayer
from .modules.agent_encoder import AgentEncoder
from .modules.map_encoder import MapEncoder
from .modules.trajectory_decoder import TrajectoryDecoder

import numpy as np

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
        state_dropout=0.75,
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
        self.num_modes = num_modes

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
        # self.agent_predictor = build_mlp(dim, [dim * 2, future_steps * 2], norm="ln")

        self.scenario_embedding = nn.Parameter(torch.randn(1, 4, dim))
        nn.init.xavier_normal_(self.scenario_embedding)

        # self.scenario_projector = ProjHead(feat_dim=dim, hidden_dim=dim//4, head_dim=8)
        self.env_projector = ProjHead(feat_dim=dim, hidden_dim=dim//4, head_dim=8)
        self.beh_projector = ProjHead(feat_dim=dim, hidden_dim=dim//4, head_dim=8)
        self.obj_projector = ProjHead(feat_dim=dim, hidden_dim=dim//4, head_dim=8)

        self.scene_target_projector = ProjHead(feat_dim=dim*2, hidden_dim=dim//4, head_dim=8)
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

        x_agent = self.agent_encoder(data)
        x_polygon = self.map_encoder(data)

        # add learnable initial scenario embedding
        scenario_emb = self.scenario_embedding.repeat(bs, 1, 1)
        
        x = torch.cat([x_agent, x_polygon], dim=1) + pos_embed 
        x = torch.cat([x, scenario_emb], dim=1)
        # x: [batch, n_elem, n_dim]. n_elem is not a fixed number, it depends on the number of agents and polygons in the scene

        for blk in self.encoder_blocks:
            x = blk(x, key_padding_mask=key_padding_mask)
        x = self.norm(x)

        # predictions, probabilities = self.trajectory_decoder(x[:, 0:A], x[:, A:-4], agent_key_padding, polygon_key_padding) # x: [batch, n_elem, 128], trajectory: [batch, modal, 80, 4], probability: [batch, 6]
        polygon_sceemb_key_padding = torch.cat([polygon_key_padding, scenario_emb_key_padding], dim=-1)
        predictions, probabilities = self.trajectory_decoder(x[:, 0:A], x[:, A:], agent_key_padding, polygon_sceemb_key_padding)
        # prediction = self.agent_predictor(x[:, 1:A]).view(bs, -1, self.future_steps, 2)

        # get the projection of the scenario embedding
        scenario_emb = x[:,-1] # [B, dim]

        # get the projection of the scenario feature embeddings
        beh_proj = self.beh_projector(x[:, -2])
        env_proj = self.env_projector(x[:, -3])
        obj_proj = self.obj_projector(x[:, -4])


        out = {
            "trajectory": predictions[:, :, 0],
            "probability": probabilities,
            "prediction": predictions,
            "beh_proj": beh_proj,
            "env_proj": env_proj,
            "obj_proj": obj_proj,
        }

        most_probable_mode = probabilities.argmax(dim=-1)

        

        if self.training:
            trajectory = predictions[:, :, 0, :, torch.Tensor([0,1,5]).to(torch.long)] # [B, num_modes, timestep, states_dim]
            trajectory_embs = self.target_encoder(trajectory.view(bs, self.num_modes, -1)) # [B, num_modes, 128]
            trajectory_embs = trajectory_embs.view((-1, self.dim)).unsqueeze(0).repeat(bs, 1, 1) # [B, B*num_modes, 128]

            scene_target_emb = torch.cat([trajectory_embs, scenario_emb.unsqueeze(1).repeat(1,bs*self.num_modes,1)], dim=-1) # [B, B*num_modes, 256]
            scene_target_emb_projs = self.scene_target_projector(scene_target_emb) # [B, B*num_modes, 8]
            out["scene_plan_emb_proj"] = scene_target_emb_projs# [B, B*num_modes, 8]

            target_emb = self.target_encoder(data["agent"]["target"][:,0].reshape(bs, -1))
            scene_target_emb = torch.cat([target_emb, scenario_emb], dim=-1)
            scene_target_emb_proj = self.scene_target_projector(scene_target_emb)
            out["scene_target_emb_proj"] = scene_target_emb_proj # [B, 8]

            ego_target = data["agent"]["target"][:, 0]
            errors = (ego_target[:, None, :, :2] - predictions[:, :, 0, :, :2]).norm(dim=-1).sum(dim=(-1))
            closest_mode = errors.argmin(dim=-1)

            output_trajectory = predictions[:, :, 0][torch.arange(bs), closest_mode]
            best_emb = self.target_encoder(output_trajectory[..., torch.Tensor([0,1,5]).to(torch.long)].reshape(bs, -1))
            scene_best_emb = torch.cat([best_emb, scenario_emb], dim=-1)
            scene_best_emb_proj = self.scene_target_projector(scene_best_emb)
            out["scene_best_emb_proj"] = scene_best_emb_proj


        if not self.training:
            trajectory = predictions[:, :, 0]
            
            output_trajectory = trajectory[torch.arange(bs), most_probable_mode]
            output_trajectory = output_trajectory[..., torch.Tensor([0,1,5]).to(torch.long)] # [B, timestep, states_dim]
            out["output_trajectory"] = output_trajectory

        return out
