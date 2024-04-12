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
from .modules.agent_encoder import AgentEncoder, EgoEncoder, TempoNet
from .modules.map_encoder import MapEncoder
from .modules.trajectory_decoder import TrajectoryDecoder
import torch.nn.functional as F

from .modules.transformer_blocks import Block
from .pretrain_model import PretrainModel
from torch.nn.utils.rnn import pad_sequence
from einops import rearrange
from .layers.embedding import Projector

from enum import Enum
import math


# no meaning, required by nuplan
trajectory_sampling = TrajectorySampling(num_poses=8, time_horizon=8, interval_length=1)

class Stage(Enum):
    PRETRAIN_SEP = 0
    PRETRAIN_MIX = 1
    PRETRAIN_REPRESENTATION = 2
    FINE_TUNING = 3

class PositionalEncoding(nn.Module):
    '''
    Standard positional encoding.
    '''
    def __init__(self, d_model, dropout=0.1, max_len=20):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        '''
        :param x: must be (T, B, H)
        :return:
        '''
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class PlanningModel(TorchModuleWrapper):
    def __init__(
        self,
        dim=128,
        state_channel=7,
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
        total_epochs=30,
        lane_mask_ratio=0.5,
        trajectory_mask_ratio=0.7,
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
        
        self.polygon_channel = polygon_channel # the number of features for each lane segment besides points coords which we will use

        self.lane_mask_ratio = lane_mask_ratio
        self.trajectory_mask_ratio = trajectory_mask_ratio

        self.no_lane_segment_points = 20

        # modules begin
        self.pe = PositionalEncoding(dim, dropout=0.1, max_len=1000)
        self.pos_emb = build_mlp(4, [dim] * 2)

        self.tempo_net = TempoNet(
            state_channel=state_channel,
            depth=3,
            num_head=8,
            dim_head=64,
        )
        

        self.agent_seed = nn.Parameter(torch.randn(dim))

        self.agent_projector = Projector(dim=dim, in_channels=self.state_channel)
        self.map_projector = Projector(dim=dim, in_channels=self.no_lane_segment_points*2+polygon_channel)
        self.lane_pred = build_mlp(dim, [512, self.no_lane_segment_points*2])
        self.agent_frame_pred = build_mlp(64, [512, 3])

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

        # self.pretrain_model = PretrainModel(
        #             embed_dim=dim,
        #             encoder_depth=encoder_depth,
        #             decoder_depth=4,
        #             num_heads=num_heads,
        #             mlp_ratio=mlp_ratio,
        #             qkv_bias=qkv_bias,
        #             drop_path=drop_path,
        #             actor_mask_ratio=0.5,
        #             lane_mask_ratio=0.5,
        #             history_steps=history_steps,
        #             future_steps=future_steps,
        #             loss_weight=[1.0, 1.0, 0.35],
        #         )
        # self.pretrain_model.initialize_weights()

        # self.decoder_norm = nn.LayerNorm(dim)

        # self.ego_decoding_token = nn.Parameter(torch.Tensor(1, 1, dim))

        

        self.trajectory_decoder = TrajectoryDecoder(
            embed_dim=dim,
            num_modes=num_modes,
            future_steps=future_steps,
            out_channels=4,
        )

        self.agent_predictor = build_mlp(dim, [dim * 2, future_steps * 2], norm="ln")

        self.apply(self._init_weights)


        # self.agent_encoder_hist = AgentEncoder(
        #     state_channel=state_channel,
        #     history_channel=history_channel,
        #     dim=dim,
        #     drop_path=drop_path,
        #     use_ego_history=True,
        #     state_attn_encoder=state_attn_encoder,
        #     state_dropout=state_dropout,
        #     starting_step=0,
        #     ending_step=self.history_steps,
        # )

        # self.agent_encoder_fut = AgentEncoder(
        #     state_channel=state_channel,
        #     history_channel=history_channel,
        #     dim=dim,
        #     drop_path=drop_path,
        #     use_ego_history=True,
        #     state_attn_encoder=state_attn_encoder,
        #     state_dropout=state_dropout,
        #     starting_step=self.history_steps,
        #     ending_step=-1,
        # )

        # self.map_encoder = MapEncoder(
        #     dim=dim,
        #     polygon_channel=polygon_channel,
        # )

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

    def get_pretrain_modules(self):
        # targeted_list = [self.pos_emb, self.tempo_net, self.agent_seed, self.agent_projector, 
        #         self.map_projector, self.lane_pred, self.agent_frame_pred, self.blocks, self.norm]
        # module_list = [[name, module] for name, module in self.named_modules() if module in targeted_list]
        return [self.pos_emb, self.tempo_net, self.agent_seed, self.agent_projector, 
                self.map_projector, self.lane_pred, self.agent_frame_pred, self.blocks, self.norm]

    def get_finetune_modules(self):
        return [self.trajectory_decoder, self.agent_predictor]


    def get_stage(self, current_epoch):
        if current_epoch < 10:
            return Stage.PRETRAIN_SEP
        # elif current_epoch < 20:
        #     return Stage.PRETRAIN_MIX
        # elif current_epoch < 25:
        #     return Stage.PRETRAIN_REPRESENTATION
        # else:
        else:
            return Stage.FINE_TUNING

    def forward(self, data, current_epoch):
        stage = self.get_stage(current_epoch)
        if stage == Stage.PRETRAIN_SEP:
            return self.forward_pretrain_separate(data)
        elif stage == Stage.PRETRAIN_MIX:
            return self.forward_pretrain_mix(data)
        elif stage == Stage.PRETRAIN_REPRESENTATION:
            return self.forward_pretrain_representation(data)
        else:
            return self.forward_fine_tuning(data)

    @staticmethod
    def extract_map_feature(data):
        # TODO: put property to one-hot?
        # TODO: split them into finer segments?
        # B M 3
        # polygon_center = data["map"]["polygon_center"]  

        # B M
        polygon_type = data["map"]["polygon_type"].long()
        polygon_on_route = data["map"]["polygon_on_route"].long()
        polygon_tl_status = data["map"]["polygon_tl_status"].long()
        # polygon_has_speed_limit = data["map"]["polygon_has_speed_limit"]
        # polygon_speed_limit = data["map"]["polygon_speed_limit"]

        # # B M 3 20 2
        point_position = data["map"]["point_position"]
        # point_vector = data["map"]["point_vector"]  

        # # B M 3 20
        # point_orientation = data["map"]["point_orientation"]

        # # B M 20
        valid_mask = data["map"]["valid_mask"]

        point_position_feature = torch.zeros(point_position[:,:,0].shape) # B M 20 2
        point_position_feature[valid_mask] = point_position[valid_mask]
        point_position_feature = rearrange(point_position_feature, 'b m p c -> b m (p c)')

        pror_feature = torch.stack([polygon_type, polygon_on_route, polygon_tl_status], dim=-1)
        feature = torch.cat([pror_feature, point_position_feature], dim=-1)

        return feature, valid_mask


    def extract_agent_feature(self, data):
        # B A T 2
        position = data["agent"]["position"][:, :, self.history_steps - 1]
        velocity = data["agent"]["velocity"][:, :, self.history_steps - 1]
        shape = data["agent"]["shape"][:, :, self.history_steps - 1]

        # B A T
        heading = data["agent"]["heading"][:, :, self.history_steps - 1]
        valid_mask = data["agent"]["valid_mask"][:, :, self.history_steps - 1]

        # B A
        category = data["agent"]["category"].long()

        frame_feature = torch.cat([position, heading.unsqueeze(-1), velocity, shape], dim=-1)
        frame_feature = rearrange(frame_feature, 'b a t c -> b a (t c)')
        feature = torch.cat([category.unsqueeze(-1), frame_feature], dim=-1)

        return feature, valid_mask
        


    @staticmethod
    def lane_random_masking(x, future_mask_ratio, key_padding_mask):
        '''
        x: (B, N, D). In the dimension D, the first two elements indicate the start point of the lane segment
        future_mask_ratio: float

        note that following the scheme of SEPT, all the attributes of the masked lane segments
        are set to zero except the starting point

        this modified version keeps the original order of the lane segments and thus can use the original key_padding_mask
        '''
        num_tokens = (~key_padding_mask).sum(1)  # (B, )
        len_keeps = torch.ceil(num_tokens * (1 - future_mask_ratio)).int()

        x_masked_list, new_key_padding_mask, ids_keep_list = [], [], []
        for i, (num_token, len_keep) in enumerate(zip(num_tokens, len_keeps)):
            noise = torch.rand(num_token, device=x.device)
            ids_shuffle = torch.argsort(noise)

            ids_keep = ids_shuffle[:len_keep]
            ids_keep_list.append(ids_keep)

            ids_masked = ids_shuffle[len_keep:]

            x_masked= x[i].clone()
            x_masked[ids_masked, 5:] = 0 # NOTE: keep polygon_type, polygon_on_route, polygon_tl_status, and the coords of starting point

            x_masked_list.append(x_masked)
            # new_key_padding_mask.append(torch.zeros(len_keep, device=x.device))
 
        x_masked_list = pad_sequence(x_masked_list, batch_first=True)
        # new_key_padding_mask = pad_sequence(
        #     new_key_padding_mask, batch_first=True, padding_value=True
        # )

        return x_masked_list, ids_keep_list
    

    @staticmethod
    def trajectory_random_masking(x, future_mask_ratio, key_padding_mask, seed):
        '''
        x: (B, N, D). In the dimension D, the first two elements indicate the start point of the lane segment
        future_mask_ratio: float
        seed: (D, )

        note that following the scheme of SEPT, all the attributes of the masked lane segments
        are set to zero except the starting point

        this modified version keeps the original order of the lane segments and thus can use the original key_padding_mask
        '''
        num_tokens = (~key_padding_mask).sum(1)  # (B, )
        len_keeps = torch.ceil(num_tokens * (1 - future_mask_ratio)).int()

        x_masked_list, new_key_padding_mask, ids_keep_list = [], [], []
        for i, (num_token, len_keep) in enumerate(zip(num_tokens, len_keeps)):
            noise = torch.rand(num_token, device=x.device)
            ids_shuffle = torch.argsort(noise)

            ids_keep = ids_shuffle[:len_keep]
            ids_keep_list.append(ids_keep)

            ids_masked = ids_shuffle[len_keep:]

            x_masked= x[i].clone()
            x_masked[ids_masked] = seed.unsqueeze(0) # NOTE: keep polygon_type, polygon_on_route, polygon_tl_status, and the coords of starting point

            x_masked_list.append(x_masked)
            # new_key_padding_mask.append(torch.zeros(len_keep, device=x.device))
 
        x_masked_list = pad_sequence(x_masked_list, batch_first=True)
        # new_key_padding_mask = pad_sequence(
        #     new_key_padding_mask, batch_first=True, padding_value=True
        # )

        return x_masked_list, ids_keep_list
    

    def forward_pretrain_separate(self, data):

        ## 1. MRM
        map_features, polygon_mask = self.extract_map_feature(data)
        polygon_key_padding = ~(polygon_mask.any(-1))

        (
            lane_masked_tokens,
            lane_ids_keep_list,
        ) = self.lane_random_masking(
            map_features, self.lane_mask_ratio, polygon_mask
        )

        lane_embedding = self.map_projector(lane_masked_tokens)

        # transformer stack
        x = lane_embedding
        for blk in self.blocks:
            x = blk(x, key_padding_mask=polygon_key_padding)
        x = self.norm(x)

        # lane pred loss
        lane_pred_mask = ~polygon_mask
        for i, idx in enumerate(lane_ids_keep_list):
            lane_pred_mask[i, idx] = False

        lane_pred = self.lane_pred(x)
        # lane_reg_mask = ~polygon_mask
        # lane_reg_mask[~lane_pred_mask] = False
        lane_pred_loss = F.mse_loss(
            lane_pred[lane_pred_mask], map_features[lane_pred_mask, 5:]
        )

        ## 2. MTM

        agent_features, agent_mask = self.extract_agent_feature(data)
        agent_key_padding = ~(agent_mask.any(-1))

        agent_embedding = self.agent_projector(agent_features) # B A D
        (agent_masked_tokens, agent_ids_keep_list) = self.trajectory_random_masking(agent_embedding, self.trajectory_mask_ratio, agent_mask, seed=self.agent_seed)

        positional_embedding = self.pe(agent_masked_tokens)

        y = self.tempo_net(agent_masked_tokens+positional_embedding, agent_key_padding)
        y = self.norm(y)

        # frame pred loss
        frame_pred = self.agent_frame_pred(y)
        agent_pred_mask = ~polygon_mask  
        for i, idx in enumerate(agent_ids_keep_list):
            agent_pred_mask[i, idx] = False
        agent_pred_loss = F.mse_loss(
            frame_pred[agent_pred_mask], agent_features[agent_pred_mask]
        )
 

        out = {
            "MRM_loss": lane_pred_loss,
            "MTM_loss": agent_pred_loss,
            "loss": lane_pred_loss + agent_pred_loss,
        }

        return out


    def forward_fine_tuning(self, data):
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

        # agent embedding
        agent_features, agent_mask = self.extract_agent_feature(data)
        agent_embedding = self.agent_projector(agent_features) # B A D
        positional_embedding = self.pe(agent_embedding)
        x_agent = self.tempo_net(agent_embedding+positional_embedding, agent_key_padding)

        # map embedding
        map_features, polygon_mask = self.extract_map_feature(data)
        lane_embedding = self.map_projector(map_features)

        x = torch.cat([x_agent, lane_embedding], dim=1) + pos_embed

        for blk in self.encoder_blocks:
            x = blk(x, key_padding_mask=key_padding_mask)
        x = self.norm(x)

        trajectory, probability = self.trajectory_decoder(x[:, 0])
        prediction = self.agent_predictor(x[:, 1:A]).view(bs, -1, self.future_steps, 2)

        out = {
            "trajectory": trajectory,
            "probability": probability,
            "prediction": prediction,
        }

        if not self.training:
            best_mode = probability.argmax(dim=-1)
            output_trajectory = trajectory[torch.arange(bs), best_mode]
            angle = torch.atan2(output_trajectory[..., 3], output_trajectory[..., 2])
            out["output_trajectory"] = torch.cat(
                [output_trajectory[..., :2], angle.unsqueeze(-1)], dim=-1
            )

        return out

    def forward_pretrain_representation(self, data):
        raise NotImplementedError
    
    def forward_pretrain_mix(self, data):
        raise NotImplementedError
        # # data preparation
        # agent_pos = data["agent"]["position"][:, :, self.history_steps - 1]
        # agent_heading = data["agent"]["heading"][:, :, self.history_steps - 1]
        # hist_mask = data["agent"]["valid_mask"][:, :, : self.history_steps]
        # fut_mask = data["agent"]["valid_mask"][:, :, self.history_steps:]
        # polygon_center = data["map"]["polygon_center"]
        # point_position = data["map"]["point_position"]
        # polygon_mask = data["map"]["valid_mask"]

        # bs, A = agent_pos.shape[0:2]

        # # positional embedding
        # position = torch.cat([agent_pos, agent_pos, polygon_center[..., :2]], dim=1)
        # angle = torch.cat([agent_heading, agent_heading, polygon_center[..., 2]], dim=1)
        # pos_feat = torch.cat(
        #     [position, torch.stack([angle.cos(), angle.sin()], dim=-1)], dim=-1
        # )
        # pos_embed = self.pos_emb(pos_feat)

        # # type information embedding
        # agent_category = data["agent"]["category"] # 4 possible types
        # polygon_type = data["map"]["polygon_type"]+4 # 3 possible types

        # '''
        # self.interested_objects_types = [
        #     TrackedObjectType.EGO,
        #     TrackedObjectType.VEHICLE,
        #     TrackedObjectType.PEDESTRIAN,
        #     TrackedObjectType.BICYCLE,
        # ]
        # self.polygon_types = [
        #     SemanticMapLayer.LANE,
        #     SemanticMapLayer.LANE_CONNECTOR,
        #     SemanticMapLayer.CROSSWALK,
        # ]
        # '''

        # # create pose embedding for each element
        # types = torch.cat([agent_category, agent_category, polygon_type], dim=1).to(torch.long)
        # type_emb_input=torch.nn.functional.one_hot(types, num_classes=7)
        # types_embedding = self.element_type_embedding(type_emb_input.to(torch.float32))

        # hist_feat = self.agent_encoder_hist(data) + pos_embed[:, :A]
        # lane_feat = self.map_encoder(data) + pos_embed[:, 2*A:]

        # # add types embedding to the input
        # hist_feat += types_embedding[:, :A]
        # lane_feat += types_embedding[:, 2*A:]

        # agent_key_padding = ~(hist_mask.any(-1))
        # polygon_key_padding = ~(polygon_mask.any(-1))
        # key_padding_mask = torch.cat([agent_key_padding, polygon_key_padding], dim=-1)

        # x_ego = self.ego_encoder(data["current_state"][:, : self.state_channel])
        # x = torch.cat([x_ego.unsqueeze(1), hist_feat[:, 1:], lane_feat], dim=1)
        # # x = torch.cat([hist_feat, lane_feat], dim=1)
        
        # for blk in self.blocks:
        #     x = blk(x, key_padding_mask=key_padding_mask)
        # x = self.norm(x)

        # x_decoder = torch.cat([self.ego_decoding_token.expand(bs, -1, -1), x], dim=1)
        # decoder_key_padding_mask = torch.cat([torch.zeros(bs, 1, dtype=torch.bool, device=x.device), key_padding_mask], dim=-1)
        # for blk in self.decoder_blocks:
        #     x_decoder = blk(x_decoder, key_padding_mask=decoder_key_padding_mask)

        # trajectory, probability = self.trajectory_decoder(x_decoder[:, 0])
        # prediction = self.agent_predictor(x_decoder[:, 2:A+1]).view(bs, -1, self.future_steps, 2)

        # out = {
        #     "trajectory": trajectory,
        #     "probability": probability,
        #     "prediction": prediction,
        # }

        # if self.training:

        #     future_feat = self.agent_encoder_fut(data) + pos_embed[:, A:2*A]
        #     future_feat += types_embedding[:, A:2*A]

        #     lane_normalized = point_position[:, :, 0] - polygon_center[..., None, :2]

        #     hist_target = data["agent"]["position"][:, :, :self.history_steps] - agent_pos[:, :, None, :]
        #     fut_target = data["agent"]["position"][:, :, self.history_steps:] - agent_pos[:, :, None, :]

        #     pretrained_out = self.pretrain_model(hist_feat, lane_feat, future_feat, hist_mask, fut_mask, polygon_key_padding, pos_feat,
        #             lane_normalized, hist_target, fut_target, types_embedding, pretrain_progress)
            
        #     pretrained_out["pretrain_loss"] = pretrained_out.pop("loss")
        #     out.update(pretrained_out)

        # else:
        #     best_mode = probability.argmax(dim=-1)
        #     output_trajectory = trajectory[torch.arange(bs), best_mode]
        #     angle = torch.atan2(output_trajectory[..., 3], output_trajectory[..., 2])
        #     out["output_trajectory"] = torch.cat(
        #         [output_trajectory[..., :2], angle.unsqueeze(-1)], dim=-1
        #     )

        #     pretrained_out={
        #         "pretrain_loss": torch.tensor(0.0),
        #         "hist_loss": torch.tensor(0.0),
        #         "future_loss": torch.tensor(0.0),
        #         "lane_pred_loss": torch.tensor(0.0),
        #         "hist_rec_pred_loss": torch.tensor(0.0),
        #         "fut_rec_pred_loss": torch.tensor(0.0),
        #         "lane_rec_pred_loss": torch.tensor(0.0),
        #         "hard_ratio": torch.tensor(0.0),
        #     }
        #     out.update(pretrained_out)

        # return out
    
    
    
    # def initialize_finetune(self):
    #     for param_pre, param_plan in zip(self.pretrain_model.blocks.parameters(), self.blocks.parameters()):
    #         param_plan.data = param_pre.data.clone().detach()
    #     for param_pre, param_plan in zip(self.pretrain_model.decoder_blocks.parameters(), self.decoder_blocks.parameters()):
    #         param_plan.data = param_pre.data.clone().detach()
    #     for param_pre, param_plan in zip(self.pretrain_model.norm.parameters(), self.norm.parameters()):
    #         param_plan.data = param_pre.data.clone().detach()
    #     for param_pre, param_plan in zip(self.pretrain_model.decoder_norm.parameters(), self.decoder_norm.parameters()):
    #         param_plan.data = param_pre.data.clone().detach()
