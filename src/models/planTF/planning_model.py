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
from .modules.adversarial_modules import TransformerMasker, AdversarialEmbeddingPerturbator

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

        self.masker = TransformerMasker(in_dim=dim, num_heads=16, mask_rate=0.7, dropout=0.1)
        self.adv_embedding_offset = AdversarialEmbeddingPerturbator(dim=dim)

        # self.inf_query_generator = TransformerEncoderLayer(dim=dim, num_heads=num_heads, drop_path=0.2)
        # self.learned_seed = nn.Parameter(torch.randn(1, 1, dim))

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
        key_padding_mask = torch.cat([agent_key_padding, polygon_key_padding], dim=-1)

        x_agent = self.agent_encoder(data)
        x_polygon = self.map_encoder(data)

        x = torch.cat([x_agent, x_polygon], dim=1) + pos_embed 
        
        # x: [batch, n_elem, n_dim]. n_elem is not a fixed number, it depends on the number of agents and polygons in the scene

        masks_sup = self.masker(x.detach(), key_padding_mask) # [batch, n_elem]
        offsets = self.adv_embedding_offset(x.detach()) # [batch, n_elem, n_dim]

        # note that the masks here are not bool, but float. it should be multiplied with the tensors to approximate the masked tensor
        # during training. 
        # reference: https://arxiv.org/pdf/1802.07814.pdf

        x_perturbed = x + offsets * masks_sup.unsqueeze(-1)

        # forward pass of the unperturbed inputs
        for blk in self.encoder_blocks:
            x = blk(x, key_padding_mask=key_padding_mask)
        x = self.norm(x)

        # forward pass of the unperturbed inputs
        x_p = x_perturbed
        for blk in self.encoder_blocks:
            x_p = blk(x_p, key_padding_mask=key_padding_mask)
        x_p = self.norm(x_p)


        predictions_full, probability_full = self.trajectory_decoder(x[:, 0:A], x[:, A:], agent_key_padding, polygon_key_padding)
        predictions_sup, probability_sup = self.trajectory_decoder(x_p[:, 0:A], x_p[:, A:], agent_key_padding, polygon_key_padding)
        trajectory_full = predictions_full[:,:,0]
        trajectory_sup = predictions_sup[:,:,0]

        out = {
            "trajectory_per": trajectory_sup,
            "probability_per": probability_sup,
            "predictions_per": predictions_sup,
            "trajectory": trajectory_full,
            "probability": probability_full,
            "predictions": predictions_full,
        }

        if not self.training:
            best_mode = probability_full.argmax(dim=-1)
            output_trajectory = trajectory_full[torch.arange(bs), best_mode]
            angle = torch.atan2(output_trajectory[..., 6], output_trajectory[..., 5])
            out["output_trajectory"] = torch.cat(
                [output_trajectory[..., :2], angle.unsqueeze(-1)], dim=-1
            )

        return out
    
    # def get_loss_final_modules(self):
    #     modules = []
    #     for module_name, module in self.named_modules():
    #         if not (module_name.startswith("sup_trajectory_decoder") or module_name.startswith("adv_masker")):
    #             modules.append((module_name, module))

    #     return modules