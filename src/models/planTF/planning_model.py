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
        mask_rate=0.3,
    ) -> None:
        super().__init__(
            feature_builders=[feature_builder],
            target_builders=[EgoTrajectoryTargetBuilder(trajectory_sampling)],
            future_trajectory_sampling=trajectory_sampling,
        )

        self.dim = dim
        self.history_steps = history_steps
        self.future_steps = future_steps
        self.encoder_depth = encoder_depth
        self.mask_rate = mask_rate

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

        self.encoder_blocks_prior = nn.ModuleList(
            TransformerEncoderLayer(dim=dim, num_heads=num_heads, drop_path=dp)
            for dp in [x.item() for x in torch.linspace(0, drop_path/2, encoder_depth//2)]
        )

        self.encoder_blocks_latter = nn.ModuleList(
            TransformerEncoderLayer(dim=dim, num_heads=num_heads, drop_path=dp)
            for dp in [x.item() for x in torch.linspace(drop_path/2, drop_path, encoder_depth//2)]
        )

        self.reconstruct_blocks_latter = nn.ModuleList(
            TransformerEncoderLayer(dim=dim, num_heads=num_heads, drop_path=dp)
            for dp in [x.item() for x in torch.linspace(drop_path/2, drop_path, encoder_depth//2)]
        )

        # self.norm_before_perturb = nn.LayerNorm(dim, elementwise_affine=False)
        self.norm_enc = nn.LayerNorm(dim)

        self.trajectory_decoder_full = MlpTrajectoryDecoder(
            embed_dim=dim,
            num_modes=num_modes,
            future_steps=future_steps,
            out_channels=4,
        )

        # self.trajectory_decoder_per = MlpTrajectoryDecoder(
        #     embed_dim=dim,
        #     num_modes=num_modes,
        #     future_steps=future_steps,
        #     out_channels=4,
        # )

        self.element_type_embedding = nn.Embedding(7, dim)
        self.masked_embedding_offset = nn.Parameter(torch.randn(1, 1, dim).to('cuda'), requires_grad=True)

        self.noise_distributor = NoiseDistributor(in_dim=dim, num_heads=16, mask_rate=0.7, dropout=0.1)
        # self.adv_embedding_offset = AdversarialEmbeddingPerturbator(dim=dim)

        # self.inf_query_generator = TransformerEncoderLayer(dim=dim, num_heads=num_heads, drop_path=0.2)
        # self.learned_seed = nn.Parameter(torch.randn(1, 1, dim))

        self.agent_predictor = build_mlp(dim, [dim * 2, future_steps * 2], norm="ln")

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
        x_initial = x.clone().detach()        
        # x: [batch, n_elem, n_dim]. n_elem is not a fixed number, it depends on the number of agents and polygons in the scene

        # forward pass of the unperturbed inputs
        for blk in self.encoder_blocks_prior:
            x = blk(x, key_padding_mask=key_padding_mask)
        for blk in self.encoder_blocks_latter:
            x = blk(x, key_padding_mask=key_padding_mask)
        x = self.norm_enc(x)

        # generate masked version of the inputs
        M = self.get_random_masks(key_padding_mask)
        x_p = (x_initial*(~M)+ (pos_embed+self.masked_embedding_offset)*M)*(~key_padding_mask.unsqueeze(-1)) 

        for blk in self.encoder_blocks_prior:
            x_p = blk(x_p, key_padding_mask=key_padding_mask)
        for blk in self.reconstruct_blocks_latter:
            x_p = blk(x_p, key_padding_mask=key_padding_mask)
        x_p = self.norm_enc(x_p)

        rec_loss = torch.norm((x_initial-x_p)*M*(~key_padding_mask.unsqueeze(-1)), dim=-1).mean()

        trajectory_full, probability_full = self.trajectory_decoder_full(x[:, 0])
        prediction_full = self.agent_predictor(x[:, 1:A]).view(bs, -1, self.future_steps, 2)

        # trajectory_per, probability_per = self.trajectory_decoder_per(x_p[:, 0])
        # prediction_per = self.agent_predictor(x_p[:, 1:A]).view(bs, -1, self.future_steps, 2)

        out = {
            # "trajectory_per": trajectory_per,
            # "probability_per": probability_per,
            # "prediction_per": prediction_per,
            "trajectory": trajectory_full,
            "probability": probability_full,
            "prediction": prediction_full,
            "rec_loss": rec_loss,
        }

        if not self.training:
            best_mode = probability_full.argmax(dim=-1)
            output_trajectory = trajectory_full[torch.arange(bs), best_mode]
            angle = torch.atan2(output_trajectory[..., 3], output_trajectory[..., 2])
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

    def get_random_masks(self, key_padding_mask):
        '''
            input: key_padding_mask: [batch, n_elem]
            output: mask: [batch, n_elem] with mask_rate of its valid elements set to 1
        '''
        padding_masks = ~key_padding_mask
        valid_num = padding_masks.sum(dim=-1)
        unmasked_num = (valid_num * self.mask_rate).to(torch.long)
        randperms = torch.stack([torch.randperm(padding_masks.shape[-1])+1 for _ in range(padding_masks.shape[0])], dim=0)
        randperms = randperms.to(padding_masks.device)*padding_masks
        sorted_randperms = torch.sort(randperms, dim=-1, descending=True)[0]
        indices_split = torch.stack([sorted_randperms[i,unmasked_num[i]] for i in range(padding_masks.shape[0])])
        mask = randperms > indices_split.unsqueeze(-1)
        return mask.unsqueeze(-1)
