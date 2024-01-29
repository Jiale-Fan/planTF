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
        decoder_depth=4,
        drop_path=0.2,
        num_heads=8,
        num_modes=6,
        use_ego_history=False,
        state_attn_encoder=True,
        state_dropout=0.75,
        feature_builder: NuplanFeatureBuilder = NuplanFeatureBuilder(),
        mask_rate_t0=0.3,
        mask_rate_tf=0.9,
        keyframes_interval = 5,
        out_channels=4,
        use_attn_mask=True,
        use_memory_mask=True,
        latent_space_dim=8,
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
        self.mask_rate_t0 = mask_rate_t0
        self.mask_rate_tf = mask_rate_tf
        self.keyframes_interval = keyframes_interval
        self.num_heads = num_heads
        self.use_attn_mask = use_attn_mask
        self.use_memory_mask = use_memory_mask
        self.latent_space_dim = latent_space_dim

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

        self.map_encoder_pred = MapEncoder(
            dim=dim,
            polygon_channel=polygon_channel,
        )

        self.map_encoder_plan = MapEncoder(
            dim=dim,
            polygon_channel=polygon_channel,
        )

        self.encoder_blocks_pred = nn.ModuleList(
            TransformerEncoderLayer(dim=dim, num_heads=num_heads, drop_path=dp)
            for dp in [x.item() for x in torch.linspace(0, drop_path, encoder_depth)]
        )

        self.decoder_blocks = nn.ModuleList(
            nn.TransformerDecoderLayer(
            d_model=dim,
            nhead=num_heads,
            dim_feedforward=2048,
            dropout=dp,
            activation="relu",
            batch_first=True,
        )
            for dp in [x.item() for x in torch.linspace(0, drop_path, decoder_depth)]
        )

        self.dim_ajust_beh = nn.Linear(self.latent_space_dim, dim)
        self.dim_ajust_beh_back = nn.Linear(dim, self.latent_space_dim)

        self.behavior_generator = nn.TransformerDecoderLayer(
            d_model=dim,
            nhead=num_heads,
            dim_feedforward=2048,
            dropout=0.2,
            activation="relu",
            batch_first=True,
        )

        self.latent_var_decoder = nn.TransformerDecoderLayer(
            d_model=dim,
            nhead=num_heads,
            dim_feedforward=2048,
            dropout=0.2,
            activation="relu",
            batch_first=True,
        )

        self.gaussian_para_mlp = GaussianParamerizationMLP(dim, self.latent_space_dim, num_layers=1)

        self.norm_enc = nn.LayerNorm(dim)
        self.norm_dec = nn.LayerNorm(dim)

        self.keyframes_indices = self.get_keyframe_indices().to('cuda')
        self.num_keyframes = self.keyframes_indices.sum().to(torch.long).item()

        self.trajectory_decoder = MlpTrajectoryDecoder(
            embed_dim=dim,
            num_modes=1,
            future_steps=future_steps,
            out_channels=out_channels,
        )

        self.element_type_embedding = nn.Embedding(7, dim)


        self.keyframes_seed = nn.Parameter(torch.randn(1, 1, dim).to('cuda'), requires_grad=True)

        self.agent_predictor = TFTrajectoryDecoder(
            embed_dim=dim,
            num_modes=num_modes,
            future_steps=future_steps,
            out_channels=7,
        )
        self.agent_context_mlp = build_mlp(dim*2, [dim], norm="ln")
        self.keyframe_mlp = build_mlp(dim, [4], norm="ln")

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
        # encode the information of the agents and the map separately without cross attention
        x_agent, agent_key_padding = self.initial_agents_encoding(data)
        x_map, map_key_padding = self.initial_map_encoding(self.map_encoder_pred, data, on_route_info=False)
        bs, A = x_agent.shape[:2]
        x = torch.cat([x_agent, x_map], dim=1) # [batch, n_elem, n_dim]
        key_padding_mask = torch.cat([agent_key_padding, map_key_padding], dim=-1) # [batch, n_elem]
        x_initial = x
        # x: [batch, n_elem, n_dim]. n_elem is not a fixed number, it depends on the number of agents and polygons in the scene

        # prediction encoder: trained in pretraining stage
        for blk in self.encoder_blocks_pred:
            x = blk(x, key_padding_mask=key_padding_mask)
        x = self.norm_enc(x)

        # get the distribution of the agents' future trajectories
        prediction, probabilities = self.agent_predictor(x[:, 0:A], x[:, A:], agent_key_padding, map_key_padding)

        latents = self.sample_latent_variables(bs) # [batch_size, latent_space_dim]
        latents_emb_dim = self.dim_ajust_beh(latents) # [batch_size, dim]
        behavior_traj_emb = self.behavior_generator(tgt=latents_emb_dim, 
                                                memory=x_map.detach().clone(),  
                                                memory_key_padding_mask=map_key_padding) # [batch_size, n_elem, dim]
        behavior_trajs, _ = self.trajectory_decoder(behavior_traj_emb)
        mutual_info_loss = self.calculate_mutual_info_loss(latents, behavior_traj_emb, x_map.detach().clone(), map_key_padding)
        

        # concatenate the initial state to the context containing the prediction, take the elements corresponding to the agents
        
        concated = torch.cat([x_initial, x], dim=-1)[:, :A].detach().clone()
        context_agt = self.agent_context_mlp(concated) # [batch, n_elem, n_dim]
        # exclude the first element of the context, which is the ego
        context_agt = context_agt[:, 1:]
        agent_key_padding = agent_key_padding[:, 1:]
        key_padding_mask = key_padding_mask[:, 1:]
        context = torch.cat([context_agt, x_map.detach().clone()], dim=1) # [batch, n_elem, n_dim]

        queries = self.keyframes_seed.repeat(bs, 1, 1) # [batch, num_keyframes, n_dim]

 
        for blk in self.decoder_blocks:
            queries = blk(tgt=queries, 
                            memory=context,
                            tgt_mask=None,
                            memory_mask=None,
                            memory_key_padding_mask=key_padding_mask)
    
        final_rep = queries # [batch, keyframes, n_dim]
        # keyframes = self.keyframe_mlp(final_rep) # [batch, keyframes, 4]

        traj_emb = self.behavior_generator(tgt=final_rep, 
                                                memory=x_map.detach().clone(),  
                                                memory_key_padding_mask=map_key_padding) # [batch_size, n_elem, dim]
        trajectory, _ = self.trajectory_decoder(traj_emb)
        probability = torch.ones(bs, 1)

        # ego_emb_hist = x[:, 0] # [batch, n_dim]
        # ego_emb = torch.cat([ego_emb_hist, keyframes.view(bs, -1)], dim=-1) # [batch, dim+keyframes*4]
        # trajectory, probability = self.trajectory_decoder(ego_emb) # [batch, num_modes, future_steps, 4]
        # trajectory[:, :, self.keyframes_indices, :] = keyframes.unsqueeze(1) # [batch, num_modes, future_steps, 4]

        out = {
            # "trajectory_per": trajectory_per,
            # "probability_per": probability_per,
            # "prediction_per": prediction_per,
            "trajectory": trajectory,
            "probability": probability,
            "prediction": prediction,
            "prediction_probabilities": probabilities,
            "behavior_trajectory": behavior_trajs,
            "mutual_info_loss": mutual_info_loss,
        }

        if not self.training:
            output_trajectory = trajectory[:, 0]
            angle = torch.atan2(output_trajectory[..., 3], output_trajectory[..., 2])
            out["output_trajectory"] = torch.cat(
                [output_trajectory[..., :2], angle.unsqueeze(-1)], dim=-1
            )

        return out
    
    def sample_latent_variables(self, bs):
        '''
            return: [batch_size, latent_space_dim]
            sample from a standard normal distribution and normalize it
        '''
        z = torch.randn(bs, self.latent_space_dim, device='cuda')
        z = z / z.norm(dim=-1, keepdim=True)
        z = z.unsqueeze(1)
        return z
    
    def calculate_mutual_info_loss(self, latents, beh_traj_emb, map_emb, map_mask):
        '''
            input: latents: [batch_size, latent_space_dim]
                   beh_traj_emb: [batch_size, dim]
            return: mutual_info_loss: [1]
        '''
        back_latents_emb = self.latent_var_decoder(tgt=beh_traj_emb,
                                                   memory=map_emb.detach().clone(),
                                                memory_key_padding_mask=map_mask)
        mu, sigma = self.gaussian_para_mlp(back_latents_emb)
        dist = torch.distributions.normal.Normal(mu, sigma)
        log_prob = dist.log_prob(latents)
        mutual_info_loss = -log_prob.mean()
        return mutual_info_loss
        

    def get_keyframe_indices(self):
        '''
            return: [future_steps] with keyframes being 1 and others being 0
        '''
        indices = torch.zeros(self.future_steps, device='cuda', dtype=torch.bool)
        indices[self.keyframes_interval-1::self.keyframes_interval] = 1
        # indices[0:20:2] = 1 # TODO: to tune
        indices[:20] = 1
        return indices
    
    # def get_loss_final_modules(self):
    #     modules = []
    #     for module_name, module in self.named_modules():
    #         if not (module_name.startswith("sup_trajectory_decoder") or module_name.startswith("adv_masker")):
    #             modules.append((module_name, module))

    #     return modules
    def initial_agents_encoding(self, data):
        '''
            input: data: dictionary containing map and agent info
                   on_route_info: bool indicating whether on_route info is used
            output: x: [batch, n_elem, n_dim]
                    key_padding_mask: [batch, n_elem] with 0 for valid elements and 1 for invalid elements
        '''
        agent_pos = data["agent"]["position"][:, :, self.history_steps - 1]
        agent_heading = data["agent"]["heading"][:, :, self.history_steps - 1]
        agent_mask = data["agent"]["valid_mask"][:, :, : self.history_steps]
        agent_category = data["agent"]["category"].to(torch.long) # 4 possible types

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
        types_embedding = self.element_type_embedding(agent_category)

        pos = torch.cat(
            [agent_pos, torch.stack([agent_heading.cos(), agent_heading.sin()], dim=-1)], dim=-1
        )
        pos_embed = self.pos_emb(pos) + types_embedding
        agent_key_padding = ~(agent_mask.any(-1))
        x_agent = self.agent_encoder(data) + pos_embed
        return x_agent, agent_key_padding
    
    def initial_map_encoding(self, map_encoder, data, on_route_info):

        polygon_center = data["map"]["polygon_center"]
        polygon_mask = data["map"]["valid_mask"]
        polygon_type = data["map"]["polygon_type"].to(torch.long)+4

        types_embedding = self.element_type_embedding(polygon_type)

        pos = torch.cat(
            [polygon_center[..., :2], torch.stack([polygon_center[..., 2].cos(), polygon_center[..., 2].sin()], dim=-1)], dim=-1
        )
        pos_embed = self.pos_emb(pos) + types_embedding
        polygon_key_padding = ~(polygon_mask.any(-1))
        x_polygon = map_encoder(data, on_route_info) + pos_embed
        return x_polygon, polygon_key_padding


    def get_decoder_tgt_masks(self, bs):
        m1 = torch.tril(torch.ones(self.num_keyframes, self.num_keyframes, device='cuda'), diagonal=-1).to(torch.bool)
        # m2 = torch.triu(torch.ones(self.num_keyframes, self.num_keyframes, device='cuda'), diagonal=5).to(torch.bool)
        # mask = (~m).repeat(bs*self.num_heads, 1, 1)
        # mask = m1 | m2
        mask = m1
        return mask
    
    def get_decoder_memory_masks(self, agent_key_padding, map_key_padding, on_route_bools):
        '''
            input: agent_key_padding: [batch, n_elem_agent]
                     map_key_padding: [batch, n_elem_map], 1 indicates invalid elements
                     on_route_bools: [batch, n_elem_map], 1 indicates on_route and 0 indicates not on_route
        '''

        ls = torch.linspace(self.mask_rate_t0, self.mask_rate_tf, self.future_steps).to(self.keyframes_indices.device)
        key_ls = ls*self.keyframes_indices
        mask_rates = key_ls[key_ls>0]

        # map_mask = map_key_padding | (~on_route_bools)
        map_mask = map_key_padding
        
        batch_agent_mask = torch.stack([self.get_mask_slice(agent_key_padding, r) \
                     for r in mask_rates], dim=1)
        batch_mask = torch.cat([batch_agent_mask, map_mask.unsqueeze(1).repeat(1,self.num_keyframes,1)], dim=-1)
        batch_mask = batch_mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
        batch_mask = batch_mask.view(-1, batch_mask.shape[-2], batch_mask.shape[-1])
        return batch_mask
        
    
    def get_mask_slice(self, key_padding_mask, mask_rate):
        mask_rate = mask_rate.to(key_padding_mask.device)
        padding_masks = ~key_padding_mask
        valid_num = padding_masks.sum(dim=-1)
        unmasked_num = torch.floor((valid_num * mask_rate)).to(torch.long)
        randperms = torch.stack([torch.randperm(padding_masks.shape[-1])+1 for _ in range(padding_masks.shape[0])], dim=0)
        randperms = randperms.to(padding_masks.device)*padding_masks
        sorted_randperms = torch.sort(randperms, dim=-1, descending=True)[0]
        indices_split = torch.stack([sorted_randperms[i,min(unmasked_num[i], sorted_randperms.shape[-1]-1)] for i in range(padding_masks.shape[0])])
        mask = randperms >= indices_split.unsqueeze(-1)
        return mask
    
    def init_stage_two(self):
        # self.map_encoder_plan.load_state_dict(self.map_encoder_pred.state_dict())
        
        params_pred = self.map_encoder_pred.state_dict()
        # params_plan = self.map_encoder_plan.state_dict()
        detached = {}
        for name, param in params_pred.items():
            detached[name] = params_pred[name].detach().clone()
        self.map_encoder_plan.load_state_dict(detached)
        pass

    def freeze_trajectory_decoder(self):
        for param in self.trajectory_decoder.parameters():
            param.requires_grad = False
        pass

class GaussianParamerizationMLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=128, num_layers=2) -> None:
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.mlp = build_mlp(in_dim, [hidden_dim] * num_layers+[out_dim * 2])

    def forward(self, x):
        mu, log_sigma = self.mlp(x).chunk(2, dim=-1)
        sigma = torch.exp(log_sigma)
        return mu, sigma