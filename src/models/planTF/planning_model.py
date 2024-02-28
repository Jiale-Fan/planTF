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
from .modules.agent_encoder import AgentEncoder, EgoEncoder
from .modules.map_encoder import MapEncoder
from .modules.trajectory_decoder import TrajectoryDecoder
import torch.nn.functional as F

from .modules.transformer_blocks import Block
from .pretrain_model import PretrainModel


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
        mlp_ratio=4.0,
        qkv_bias=False,
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
        self.state_channel = state_channel


        self.pos_emb = build_mlp(4, [dim] * 2)
        self.agent_encoder_hist = AgentEncoder(
            state_channel=state_channel,
            history_channel=history_channel,
            dim=dim,
            drop_path=drop_path,
            use_ego_history=True,
            state_attn_encoder=state_attn_encoder,
            state_dropout=state_dropout,
            starting_step=0,
            ending_step=self.history_steps,
        )

        self.agent_encoder_fut = AgentEncoder(
            state_channel=state_channel,
            history_channel=history_channel,
            dim=dim,
            drop_path=drop_path,
            use_ego_history=True,
            state_attn_encoder=state_attn_encoder,
            state_dropout=state_dropout,
            starting_step=self.history_steps,
            ending_step=-1,
        )

        self.ego_encoder = EgoEncoder(
            state_channel=state_channel,
            dim=dim,
            state_dropout=state_dropout,
        )

        self.map_encoder = MapEncoder(
            dim=dim,
            polygon_channel=polygon_channel,
        )

        self.element_type_embedding = build_mlp(7, [dim] * 2, norm="ln")

        dpr = [x.item() for x in torch.linspace(0, drop_path, encoder_depth)]

        self.blocks = nn.ModuleList(
            Block(
                dim=dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop_path=dpr[i],
            )
            for i in range(encoder_depth)
        )

        self.norm = nn.LayerNorm(dim)

        self.pretrain_model = PretrainModel(
                    embed_dim=dim,
                    encoder_depth=encoder_depth,
                    decoder_depth=4,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop_path=drop_path,
                    actor_mask_ratio=0.5,
                    lane_mask_ratio=0.5,
                    history_steps=history_steps,
                    future_steps=future_steps,
                    loss_weight=[1.0, 1.0, 0.35],
                )
        self.pretrain_model.initialize_weights()

        self.trajectory_decoder = TrajectoryDecoder(
            embed_dim=dim,
            num_modes=num_modes,
            future_steps=future_steps,
            out_channels=4,
        )

        self.agent_predictor = build_mlp(dim, [dim * 2, future_steps * 2], norm="ln")


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

        # data preparation
        agent_pos = data["agent"]["position"][:, :, self.history_steps - 1]
        agent_heading = data["agent"]["heading"][:, :, self.history_steps - 1]
        hist_mask = data["agent"]["valid_mask"][:, :, : self.history_steps]
        fut_mask = data["agent"]["valid_mask"][:, :, self.history_steps:]
        polygon_center = data["map"]["polygon_center"]
        point_position = data["map"]["point_position"]
        polygon_mask = data["map"]["valid_mask"]

        bs, A = agent_pos.shape[0:2]

        # positional embedding
        position = torch.cat([agent_pos, agent_pos, polygon_center[..., :2]], dim=1)
        angle = torch.cat([agent_heading, agent_heading, polygon_center[..., 2]], dim=1)
        pos_feat = torch.cat(
            [position, torch.stack([angle.cos(), angle.sin()], dim=-1)], dim=-1
        )
        pos_embed = self.pos_emb(pos_feat)

        # type information embedding
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
        types = torch.cat([agent_category, agent_category, polygon_type], dim=1).to(torch.long)
        type_emb_input=torch.nn.functional.one_hot(types, num_classes=7)
        types_embedding = self.element_type_embedding(type_emb_input.to(torch.float32))

        hist_feat = self.agent_encoder_hist(data) + pos_embed[:, :A]
        lane_feat = self.map_encoder(data) + pos_embed[:, 2*A:]

        # add types embedding to the input
        hist_feat += types_embedding[:, :A]
        lane_feat += types_embedding[:, 2*A:]

        agent_key_padding = ~(hist_mask.any(-1))
        polygon_key_padding = ~(polygon_mask.any(-1))
        key_padding_mask = torch.cat([agent_key_padding, polygon_key_padding], dim=-1)

        x_ego = self.ego_encoder(data["current_state"][:, : self.state_channel])
        x = torch.cat([x_ego.unsqueeze(1), hist_feat[:, 1:], lane_feat], dim=1)
        
        for blk in self.blocks:
            x = blk(x, key_padding_mask=key_padding_mask)
        x = self.norm(x)

        trajectory, probability = self.trajectory_decoder(x[:, 0])
        prediction = self.agent_predictor(x[:, 1:A]).view(bs, -1, self.future_steps, 2)

        out = {
            "trajectory": trajectory,
            "probability": probability,
            "prediction": prediction,
        }

        if self.training:

            future_feat = self.agent_encoder_fut(data) + pos_embed[:, A:2*A]
            future_feat += types_embedding[:, A:2*A]

            lane_normalized = point_position[:, :, 0] - polygon_center[..., None, :2]

            hist_target = data["agent"]["position"][:, :, :self.history_steps] - agent_pos[:, :, None, :]
            fut_target = data["agent"]["position"][:, :, self.history_steps:] - agent_pos[:, :, None, :]

            pretrained_out = self.pretrain_model(hist_feat, lane_feat, future_feat, hist_mask, fut_mask, polygon_key_padding, pos_feat,
                    lane_normalized, hist_target, fut_target, types_embedding)
            
            pretrained_out["pretrain_loss"] = pretrained_out.pop("loss")
            out.update(pretrained_out)

        else:
            best_mode = probability.argmax(dim=-1)
            output_trajectory = trajectory[torch.arange(bs), best_mode]
            angle = torch.atan2(output_trajectory[..., 3], output_trajectory[..., 2])
            out["output_trajectory"] = torch.cat(
                [output_trajectory[..., :2], angle.unsqueeze(-1)], dim=-1
            )

            pretrained_out={
                "pretrain_loss": torch.tensor(0.0),
                "hist_loss": torch.tensor(0.0),
                "future_loss": torch.tensor(0.0),
                "lane_pred_loss": torch.tensor(0.0),
            }
            out.update(pretrained_out)

        return out
    
    def initialize_finetune(self):
        for param_pre, param_plan in zip(self.pretrain_model.blocks.parameters(), self.blocks.parameters()):
            param_plan.data = param_pre.data.clone().detach()
