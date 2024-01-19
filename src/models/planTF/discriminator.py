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
from .modules.trajectory_decoder import TFTrajectoryDecoder, MlpTrajectoryDecoder
from .modules.adversarial_modules import NoiseDistributor

# no meaning, required by nuplan
trajectory_sampling = TrajectorySampling(num_poses=8, time_horizon=8, interval_length=1)


class Discriminator(nn.Module):
    def __init__(
        self,
        dim=128,
        state_channel=6,
        polygon_channel=6,
        history_channel=9,
        total_steps=101,
        history_step=80,
        encoder_depth=4,
        drop_path=0.2,
        num_heads=8,
    ) -> None:
        super().__init__()
        

        self.dim = dim
        self.total_steps = total_steps
        self.encoder_depth = encoder_depth
        self.history_steps = history_step

        self.pos_emb = build_mlp(4, [dim] * 2)
        self.agent_encoder = AgentEncoder(
            state_channel=state_channel,
            history_channel=history_channel,
            dim=dim,
            hist_steps=total_steps,
            drop_path=drop_path,
            use_ego_history=True,
            state_attn_encoder=True,
            state_dropout=0,
        )

        self.map_encoder = MapEncoder(
            dim=dim,
            polygon_channel=polygon_channel,
        )

        self.encoder_blocks_prior = nn.ModuleList(
            TransformerEncoderLayer(dim=dim, num_heads=num_heads, drop_path=dp)
            for dp in [x.item() for x in torch.linspace(0, drop_path/2, encoder_depth//2)]
        )

        self.encoder_blocks_latter = nn.ModuleList(
            TransformerEncoderLayer(dim=dim, num_heads=num_heads, drop_path=dp)
            for dp in [x.item() for x in torch.linspace(drop_path/2, drop_path, encoder_depth//2)]
        )

        self.element_type_embedding = nn.Embedding(7, dim)
        self.norm_enc = nn.LayerNorm(dim)

        self.lin = nn.Linear(dim, 1)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm) and m.elementwise_affine:
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.2)

    def forward(self, data):

        agent_pos = data["agent"]["position"][:, :, self.history_steps - 1]
        agent_heading = data["agent"]["heading"][:, :, self.history_steps - 1]
        agent_mask = data["agent"]["valid_mask"][:, :, : self.history_steps]
        polygon_center = data["map"]["polygon_center"]
        polygon_mask = data["map"]["valid_mask"]

        agent_category = data["agent"]["category"] # 4 possible types
        polygon_type = data["map"]["polygon_type"]+4 # 3 possible types

        '''
        self.interested_objects_types = [
            TrackedObjectType.EGO,
            TrackedObjectType.VEHICLE,
            TrackedObjectType.PEDESTRIAN,
            TrackedObjectType.BICYCLE,
        ]
        self.polygon_types = [
            SemanticMapLayer.LANE,
            SemanticMapLayer.LANE_CONNECTOR,
            SemanticMapLayer.CROSSWALK,
        ]
        '''

        # create pose embedding for each element
        types = torch.cat([agent_category, polygon_type], dim=1).to(torch.long)
        types_embedding = self.element_type_embedding(types)

        bs, A = agent_pos.shape[0:2]

        position = torch.cat([agent_pos, polygon_center[..., :2]], dim=1)
        angle = torch.cat([agent_heading, polygon_center[..., 2]], dim=1)
        pos = torch.cat(
            [position, torch.stack([angle.cos(), angle.sin()], dim=-1)], dim=-1
        )
        pos_embed = self.pos_emb(pos) + types_embedding

        agent_key_padding = ~(agent_mask.any(-1))
        polygon_key_padding = ~(polygon_mask.any(-1))
        key_padding_mask = torch.cat([agent_key_padding, polygon_key_padding], dim=-1)

        x_agent = self.agent_encoder(data)
        x_polygon = self.map_encoder(data)

        x = torch.cat([x_agent, x_polygon], dim=1) + pos_embed 
        
        for blk in self.encoder_blocks_prior:
            x = blk(x, key_padding_mask=key_padding_mask)
        for blk in self.encoder_blocks_latter:
            x = blk(x, key_padding_mask=key_padding_mask)
        x = self.norm_enc(x)

        out = nn.functional.sigmoid(self.lin(x[:, 0]).squeeze(-1))
        return out