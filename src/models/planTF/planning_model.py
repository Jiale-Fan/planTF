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
from .modules.trajectory_decoder import MultimodalTrajectoryDecoder, SinglemodalTrajectoryDecoder
import torch.nn.functional as F

from .modules.transformer_blocks import Block, CrossAttender
from .pretrain_model import PretrainModel
from torch.nn.utils.rnn import pad_sequence
from einops import rearrange, reduce, repeat
from .layers.embedding import Projector

from enum import Enum
import math

from .debug_vis import plot_lane_segments, plot_scene_attention
from .info_distortor import InfoDistortor


# no meaning, required by nuplan
trajectory_sampling = TrajectorySampling(num_poses=8, time_horizon=8, interval_length=1)

def to_vector(feat, valid_mask):
    vec_mask = valid_mask[..., :-1] & valid_mask[..., 1:]

    while len(vec_mask.shape) < len(feat.shape):
        vec_mask = vec_mask.unsqueeze(-1)

    return torch.where(
        vec_mask,
        feat[:, :, 1:, ...] - feat[:, :, :-1, ...],
        torch.zeros_like(feat[:, :, 1:, ...]),
    )

class Stage(Enum):
    PRETRAIN_SEP = 0
    PRETRAIN_MIX = 1
    PRETRAIN_REPRESENTATION = 2
    DEFAULT = 3
    ANT_MASK_FINETUNE = 4

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_parameter('pe', nn.Parameter(pe, requires_grad=False))

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class PlanningModel(TorchModuleWrapper):
    def __init__(
        self,
        dim=256,
        state_channel=8,
        polygon_channel=3,
        history_channel=9,
        history_steps=21,
        future_steps=80,
        encoder_depth=4,
        decoder_depth=3, 
        drop_path=0.2,
        num_heads=8,
        num_modes=6,
        use_ego_history=False,
        state_attn_encoder=True,
        state_dropout=0.75,
        mlp_ratio=4.0,
        qkv_bias=False,
        total_epochs=35,
        lane_mask_ratio=0.5,
        trajectory_mask_ratio=0.7,
        # pretrain_epoch_stages = [0, 10, 20, 25, 30, 35], # SEPT, ft, ant, ft, ant, ft
        pretrain_epoch_stages = [0, 0],
        lane_split_threshold=20,
        alpha=0.999,
        expanded_dim = 2048,
        gamma = 1.0, # VICReg standard deviation target 
        out_channels = 4,
        N_mask = 2,
        waypoints_number = 20,
        whether_split_lane = False,
        ori_threshold = 0.7653,
        feature_builder: NuplanFeatureBuilder = NuplanFeatureBuilder(),
    ) -> None:
        super().__init__(
            feature_builders=[feature_builder],
            target_builders=[EgoTrajectoryTargetBuilder(trajectory_sampling)],
            future_trajectory_sampling=trajectory_sampling,
        )

        self.inference_counter = 0

        self.dim = dim
        self.history_steps = history_steps
        self.future_steps = future_steps
        self.state_channel = state_channel
        self.num_modes = num_modes
        self.waypoints_number = waypoints_number
        
        self.polygon_channel = polygon_channel # the number of features for each lane segment besides points coords which we will use

        self.lane_mask_ratio = lane_mask_ratio
        self.trajectory_mask_ratio = trajectory_mask_ratio
        self.pretrain_epoch_stages = pretrain_epoch_stages

        self.no_lane_segment_points = 20
        self.lane_split_threshold = lane_split_threshold
        self.alpha = alpha
        self.gamma = gamma
        self.expanded_dim = expanded_dim
        self.out_channels = out_channels
        self.N_mask = N_mask
        self.whether_split_lane = whether_split_lane
        self.ori_threshold = ori_threshold

        # modules begin
        self.pe = PositionalEncoding(dim, dropout=0.1, max_len=1000)
        self.pos_emb = build_mlp(4, [dim] * 2)

        self.tempo_net = TempoNet(
            state_channel=state_channel,
            depth=3,
            num_head=8,
            dim_head=dim,
            now_timestep=history_steps-2, # -1 for the difference vector, -1 for the zero-based index
        )
        

        self.TempoNet_frame_seed = nn.Parameter(torch.randn(dim))
        self.MRM_seed = nn.Parameter(torch.randn(dim))
        # self.multimodal_seed = nn.Parameter(torch.randn(num_modes, dim))
        self.ego_seed = nn.Parameter(torch.randn(dim))

        self.seed_2s_mlp = build_mlp(dim, [dim * 2 , dim], norm="ln")
        self.seed_8s_mlp = build_mlp(dim, [dim * 2 , dim], norm="ln")

        self.agent_projector = Projector(to_dim=dim, in_channels=11) # NOTE: make consistent to state_channel
        self.agent_type_emb = nn.Embedding(4, dim)
        # self.map_encoder = Projector(dim=dim, in_channels=self.no_lane_segment_points*2+5) # NOTE: make consistent to polygon_channel
        self.map_encoder = MapEncoder(
            dim=dim,
        )
        self.lane_pred = build_mlp(dim, [512, self.no_lane_segment_points*2])
        self.agent_frame_predictor = build_mlp(dim, [512, 4])

        dpr = [x.item() for x in torch.linspace(0, drop_path, encoder_depth)]

        self.SpaNet = nn.ModuleList(
            Block(
                dim=dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop_path=dpr[i],
            )
            for i in range(encoder_depth)
        )
        self.norm_spa = nn.LayerNorm(dim)

        self.WpNet = nn.ModuleList(
            Block(
                dim=dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop_path=dpr[i],
            )
            for i in range(encoder_depth)
        )
        self.norm_wp = nn.LayerNorm(dim)

        # self.expander = nn.Linear(dim*num_seeds, expanded_dim)

        # self.blocks_teacher = nn.ModuleList(
        #     Block(
        #         dim=dim,
        #         num_heads=num_heads,
        #         mlp_ratio=mlp_ratio,
        #         qkv_bias=qkv_bias,
        #         drop_path=dpr[i],
        #     )
        #     for i in range(encoder_depth)
        # )
        # self.norm_teacher = nn.LayerNorm(dim)
        # self.expander_teacher = nn.Linear(dim*out_channels, expanded_dim)
        # self.student_list = [self.blocks, self.norm, self.expander]
        # self.teacher_list = [self.blocks_teacher, self.norm_teacher, self.expander_teacher]

        # self.flag_teacher_init = False

        # self.distortor = InfoDistortor(
        #     dt=0.1,
        #     hist_len=21,
        #     low=[-1.0, -0.75, -0.35, -1, -0.5, -0.2, -0.1],
        #     high=[1.0, 0.75, 0.35, 1, 0.5, 0.2, 0.1],
        #     augment_prob=0.5,
        #     normalize=True,
        # )
        

        self.waypoint_decoder = SinglemodalTrajectoryDecoder(
            embed_dim=dim,
            future_steps=self.waypoints_number,
            out_channels=4,
        )

        self.far_future_traj_decoder = SinglemodalTrajectoryDecoder(
            embed_dim=dim,
            future_steps=self.future_steps-self.waypoints_number,
            out_channels=4,
        )

        self.FFNet = nn.ModuleList(
            Block(
                dim=dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop_path=dpr[i],
            )
            for i in range(encoder_depth)
        )
        self.norm_ff = nn.LayerNorm(dim)
        # self.coarse_to_fine_decoder = nn.ModuleList(
        #     torch.nn.TransformerDecoderLayer(dim, 
        #                                      num_heads, 
        #                                      dim_feedforward=int(mlp_ratio*dim), 
        #                                      dropout=0.1, 
        #                                      activation="gelu", 
        #                                      batch_first=True)
        #     for i in range(decoder_depth)
        # )

        # self.trajectory_mlp = build_mlp(dim, [512, future_steps * out_channels], norm=None)
        # self.score_mlp = build_mlp(dim, [512, 1], norm=None)

        # coarse to fine planning
        self.goal_mlp = build_mlp(dim, [512, out_channels], norm=None)
        self.score_mlp = nn.Sequential(
            nn.Linear(dim, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
            nn.Tanh()
        ) # since we need the last activation to be sigmoid, we do not use the build_mlp function

        self.agent_tail_predictor = build_mlp(dim, [dim * 2, (future_steps-1) * 4], norm="ln")
        self.abs_agent_predictor = build_mlp(dim, [dim * 2, future_steps * 2], norm="ln")
        self.rel_agent_predictor = build_mlp(dim, [dim * 2, waypoints_number * 2], norm="ln")
        self.lane_emb_wp_2s_mlp = build_mlp(dim, [dim * 2 , dim], norm="ln")
        self.lane_emb_ff_2s_mlp = build_mlp(dim, [dim * 2 , dim], norm="ln")
        self.lane_emb_wp_8s_mlp = build_mlp(dim, [dim * 2 , dim], norm="ln")
        self.lane_emb_ff_8s_mlp = build_mlp(dim, [dim * 2 , dim], norm="ln")
        self.waypoints_embedder = build_mlp(self.waypoints_number * 4, [dim*2, dim], norm=None)

        self.lane_intention_2s_predictor = Projector(to_dim=1, in_channels=dim)
        self.lane_intention_8s_predictor = Projector(to_dim=1, in_channels=dim)
        self.attraction_point_projector = Projector(to_dim=dim, in_channels=4)
        self.vel_token_projector = Projector(to_dim=dim, in_channels=1)

        self.cme_motion_mlp = build_mlp(dim, [2048, 256], norm="ln")
        self.cme_env_mlp = build_mlp(dim, [2048, 256], norm="ln")

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

    def get_pretrain_modules(self):
        # targeted_list = [self.pos_emb, self.tempo_net, self.agent_seed, self.agent_projector, 
        #         self.map_projector, self.lane_pred, self.agent_frame_pred, self.blocks, self.norm]
        # module_list = [[name, module] for name, module in self.named_modules() if module in targeted_list]
        return [self.pos_emb, self.tempo_net, self.TempoNet_frame_seed, self.agent_projector, self.MRM_seed,
                self.map_encoder, self.lane_pred, self.agent_frame_predictor, self.SpaNet, self.norm_spa, self.agent_tail_predictor, 
                # JointMotion CME 
                self.cme_motion_mlp, self.cme_env_mlp, self.seed_2s_mlp, self.seed_8s_mlp]

    def get_finetune_modules(self):
        return [self.ego_seed, self.waypoint_decoder, self.far_future_traj_decoder, self.FFNet, self.goal_mlp,
                self.abs_agent_predictor, self.rel_agent_predictor, self.lane_intention_2s_predictor, self.attraction_point_projector,
                self.lane_intention_8s_predictor, 
                self.vel_token_projector, self.WpNet, self.norm_wp, self.norm_ff, 
                self.lane_emb_wp_2s_mlp, self.lane_emb_ff_2s_mlp, self.lane_emb_wp_8s_mlp, self.lane_emb_ff_8s_mlp, self.score_mlp, 
                self.waypoints_embedder]


    def get_stage(self, current_epoch):
        return Stage.DEFAULT
        if current_epoch < self.pretrain_epoch_stages[1]:
            return Stage.PRETRAIN_SEP
        # elif current_epoch < self.pretrain_epoch_stages[2]:
        #     if not self.flag_teacher_init:
        #         self.initialize_teacher()
        #         self.flag_teacher_init = True
        #     return Stage.PRETRAIN_REPRESENTATION
        # elif current_epoch < self.pretrain_epoch_stages[2]:
        #     return Stage.FINETUNE
        # elif current_epoch < self.pretrain_epoch_stages[3]:
        #     return Stage.ANT_MASK_FINETUNE
        # elif current_epoch < self.pretrain_epoch_stages[4]:
        #     return Stage.FINETUNE
        # elif current_epoch < self.pretrain_epoch_stages[5]:
        #     return Stage.ANT_MASK_FINETUNE
        else:
            # return Stage.ANT_MASK_FINETUNE
            return Stage.DEFAULT
        # for debugging
        

    def forward(self, data, current_epoch=None):
        # return self.forward_multimodal_finetune(data)
        if current_epoch is None: # when inference
            # return self.forward_pretrain_separate(data)
            return self.forward_inference(data)
            # return self.forward_antagonistic_mask_finetune(data, current_epoch)
        else:
            # if self.training and current_epoch <= 10:
            #     return self.forward_CME_pretrain(data)
            if self.training and current_epoch <= 20:
                return self.forward_teacher_enforcing(data)
            elif self.training and current_epoch > 20:
                return self.forward_multimodal_finetune(data)
            else:
                return self.forward_inference(data)
            # stage = self.get_stage(current_epoch)
            # if stage == Stage.PRETRAIN_SEP:
            #     return self.forward_pretrain_separate(data)
            # elif stage == Stage.PRETRAIN_MIX:
            #     return self.forward_pretrain_mix(data)
            # elif stage == Stage.PRETRAIN_REPRESENTATION:
            #     # self.EMA_update() # currently this is done in lightning_trainer.py
            #     return self.forward_pretrain_representation(data)
            # elif stage == Stage.FINETUNE:
            #     return self.forward_finetune(data)
            # elif stage == Stage.ANT_MASK_FINETUNE:
            #     return self.forward_antagonistic_mask_finetune(data, current_epoch)
            # else:
            #     raise NotImplementedError(f"Stage {stage} is not implemented.")

    def extract_map_feature(self, data, need_route_kpmask=False):
        # NOTE: put property to one-hot?
        # the segments longer than 20m will be splited into finer segments

        # B M 3
        # polygon_center = data["map"]["polygon_center"]  

        # B M
        polygon_type = data["map"]["polygon_type"].long()
        polygon_on_route = data["map"]["polygon_on_route"].long()
        polygon_tl_status = data["map"]["polygon_tl_status"].long()
        polygon_has_speed_limit = data["map"]["polygon_has_speed_limit"]
        polygon_speed_limit = data["map"]["polygon_speed_limit"]

        polygon_property = torch.stack([polygon_type, polygon_on_route, polygon_tl_status, polygon_has_speed_limit, polygon_speed_limit], dim=-1)

        # # B M 3 20 2
        point_position = data["map"]["point_position"]
        # point_vector = data["map"]["point_vector"]  

        # # B M 3 20
        point_orientation = data["map"]["point_orientation"]

        # # B M 20
        valid_mask = data["map"]["valid_mask"]

        # point_position_feature = torch.zeros(point_position[:,:,0].shape, device=point_position.device) # B M 20 2
        # point_position_feature[valid_mask] = point_position[:,:,0][valid_mask]
        if self.whether_split_lane:
            point_position_feature, valid_mask_r, new_poly_prop = PlanningModel.split_lane_segment(point_position[:,:,0], valid_mask, self.lane_split_threshold, polygon_property)
        else:
            point_position_feature = point_position[:,:,0]
            valid_mask_r = valid_mask
            new_poly_prop = polygon_property
        
        # point_position_feature = rearrange(point_position_feature, 'b m p c -> b m (p c)')

        # feature = torch.cat([point_position_feature, new_poly_prop], dim=-1)
        polygon_key_padding = ~(valid_mask_r.any(-1))

        point_orientation_feature = point_orientation[:,:,0].unsqueeze(-1) # B M 20

        if not need_route_kpmask:
            return point_position_feature, point_orientation_feature, new_poly_prop, valid_mask_r, polygon_key_padding
        else: 
            route_kpmask = ~((polygon_key_padding==0)&(new_poly_prop[..., 1]==1)) # valid and on route, then take the opposite
            return point_position_feature, point_orientation_feature, new_poly_prop, valid_mask_r, polygon_key_padding, route_kpmask # [B, M_new]



    @staticmethod
    def split_lane_segment(point_position_feature, valid_mask, split_threshold, polygon_properties):
        '''
            point_position_feature: (B, M, P=21, D=2)
            valid_mask: (B, M, P=21)
            split_threshold: float
            polygon_property: (B, M, 5)
        '''
        # examine whether the segment's distance between starting point and ending point exceeds 5 meters.
        # if so, we split it into two segments
        B, M, P, D = point_position_feature.shape

        points_list = []
        valid_mask_list = []
        poly_property_list = []

        for i in range(B):

            points = point_position_feature[i] # (M, P, D)
            # to prevent the code to be too complicated, if the segment contains less than P points, we assume it does not need to be splited
            valid_mask_scene = valid_mask[i] # (M, P)
            poly_property = polygon_properties[i] # (M, 5)
            
            mask_valid = valid_mask_scene.any(-1) # (M,)
            mask_cal_length = valid_mask_scene.all(-1) # (M, )

            lengths = torch.norm(points[:, 0] - points[:, -1], dim=-1) # (M, )
            long_seg = lengths > split_threshold # (M, )
            seg_to_split = mask_cal_length & long_seg # (M, )

            while seg_to_split.any():

                seg_splited = rearrange(points[seg_to_split], 'n (s q) d -> (n s) d q', s=2) # this permutation is required by the interpolation function
                seg_interpolated = torch.nn.functional.interpolate(seg_splited, size=P, mode='linear') # (n s) d P
                seg_interpolated = rearrange(seg_interpolated, 'n d p -> n p d')
                
                to_keep_mask = mask_valid & (~seg_to_split)
                points = torch.cat((points[to_keep_mask], seg_interpolated), dim=0)
                valid_mask_scene = torch.cat((valid_mask_scene[to_keep_mask],
                                            torch.ones(seg_interpolated.shape[0], seg_interpolated.shape[1],
                                            device=valid_mask.device, dtype=torch.bool)), dim=0)
                poly_property = torch.cat((poly_property[to_keep_mask], poly_property[seg_to_split], poly_property[seg_to_split]), dim=0)
                
                mask_valid = valid_mask_scene.any(-1) # (M,)
                mask_cal_length = valid_mask_scene.all(-1) # (M, )

                lengths = torch.norm(points[:, 0] - points[:, -1], dim=-1) # (M, )
                long_seg = lengths > split_threshold # (M, )
                seg_to_split = mask_cal_length & long_seg # (M, )

            points_list.append(points)
            valid_mask_list.append(valid_mask_scene)
            poly_property_list.append(poly_property)

            
        new_point_position_feature = pad_sequence(points_list, batch_first=True, padding_value=False)
        new_valid_mask = pad_sequence(valid_mask_list, batch_first=True, padding_value=False)
        new_poly_property = pad_sequence(poly_property_list, batch_first=True, padding_value=0)
                    
        return new_point_position_feature, new_valid_mask, new_poly_property

 

    def extract_agent_feature(self, data, include_future=False, get_critical_points=False):

        steps = self.history_steps if not include_future else self.history_steps + self.future_steps

        # B A T 2
        position = data["agent"]["position"][:, :, :steps]
        velocity = data["agent"]["velocity"][:, :, :steps]
        shape = data["agent"]["shape"][:, :, :steps]

        # B A T
        heading = data["agent"]["heading"][:, :, :steps]
        valid_mask = data["agent"]["valid_mask"][:, :, :steps]

        # B A
        category = data["agent"]["category"].long()
        valid_mask_vec = valid_mask[..., 1:] & valid_mask[..., :-1]
        heading_vec = to_vector(heading, valid_mask)
        agent_feature = torch.cat(
            [
                position[:,:,1:], 
                torch.stack([heading_vec.cos(), heading_vec.sin()], dim=-1),
                to_vector(position, valid_mask),
                to_vector(velocity, valid_mask),
                shape[:, :, 1:],
                valid_mask_vec.float().unsqueeze(-1),
            ],
            dim=-1,
        )

        ego_state = data["current_state"] # B, 7 (x, y, yaw, vel, acc, steer, yaw_rate). actually yaw_rate can be inferred from vel and steer

        agent_key_padding = ~(valid_mask.any(-1))

        # frame_feature = torch.cat([position, heading.unsqueeze(-1), velocity, shape], dim=-1)
        # # frame_feature = rearrange(frame_feature, 'b a t c -> b a (t c)')
        # category_rep = repeat(category, 'b a -> b a t d', t=steps, d = 1)
        # feature = torch.cat([frame_feature, category_rep], dim=-1)

        ret = [agent_feature, category, valid_mask_vec, agent_key_padding, ego_state]

        if get_critical_points:
             # B 4
            attraction_point = torch.cat([data["agent"]["position"][:, 0, self.history_steps+self.waypoints_number-1], 
                                        torch.stack([data["agent"]["heading"][:, 0, self.history_steps+self.waypoints_number-1].cos(), 
                                                    data["agent"]["heading"][:, 0, self.history_steps+self.waypoints_number-1].sin()], dim=-1),], dim=-1).clone()
            
            # B 4
            horizon_point = torch.cat([data["agent"]["position"][:, 0, -1],
                                        torch.stack([data["agent"]["heading"][:, 0, -1].cos(),
                                                    data["agent"]["heading"][:, 0, -1].sin()], dim=-1),], dim=-1).clone()

            # B, 20, 4
            waypoints_gt = torch.cat([data["agent"]["position"][:, 0, self.history_steps:self.history_steps+self.waypoints_number], 
                                        torch.stack([data["agent"]["heading"][:, 0, self.history_steps:self.history_steps+self.waypoints_number].cos(), 
                                                    data["agent"]["heading"][:, 0, self.history_steps:self.history_steps+self.waypoints_number].sin()], dim=-1),], dim=-1).clone()




            ret.extend([attraction_point, horizon_point, waypoints_gt])

        return ret
        

    def lane_random_masking(self, lane_embedding, future_mask_ratio, key_padding_mask):
        '''
        x: (B, N, D). In the dimension D, the first two elements indicate the start point of the lane segment
        future_mask_ratio: float

        note that following the scheme of SEPT, all the attributes of the masked lane segments
        are set to zero except the starting point

        this modified version keeps the original order of the lane segments and thus can use the original key_padding_mask
        '''
        new_lane_embedding = lane_embedding.clone()

        num_tokens = (~key_padding_mask).sum(1)  # (B, )
        len_keeps = torch.ceil(num_tokens * (1 - future_mask_ratio)).int()

        ids_keep_list = []
        for i, (num_token, len_keep) in enumerate(zip(num_tokens, len_keeps)):
            noise = torch.rand(num_token, device=lane_embedding.device)
            ids_shuffle = torch.argsort(noise)

            ids_keep = ids_shuffle[:len_keep]
            ids_keep_list.append(ids_keep)

            ids_masked = ids_shuffle[len_keep:]

            new_lane_embedding[i, ids_masked] = self.MRM_seed

        return new_lane_embedding, ids_keep_list


    def trajectory_random_masking(self, x, future_mask_ratio, frame_valid_mask):
        '''
        x: (B, A, T, D). 
        future_mask_ratio: float
        frame_valid_mask: (B, A, T)

        each history consists of T frames, but not all frames are valid
        we first randomly masked out future_mask_ratio of the frames, but there is a possibility that all valid frames are masked
        therefore we manually give at least one valid frame to keep
        '''
        B, A, T, D = x.shape

        len_keep = torch.ceil(frame_valid_mask.sum(-1) * (1 - future_mask_ratio)).long().unsqueeze(-1) # (B, A, 1)

        noise = torch.rand(frame_valid_mask.shape[:3], device=x.device)
        noise[~frame_valid_mask] = 0 # set the invalid frames to 0
        sorted, idx = torch.sort(noise, descending=True)

        threshold = torch.gather(sorted, -1, len_keep)
        replace_mask = noise < threshold # all invalid frames are substituted with TempoNet_frame_seed, but this should not be a problem since in transformer the key_padding_mask is used

        masked_x = x.clone()
        masked_x[replace_mask] = self.TempoNet_frame_seed

        pred_mask = frame_valid_mask & replace_mask

        # for i in range(B):
        #     for j in range(A):
        #         masked_x[i, j, idx[i, j, len_keep[i, j]:]] = self.TempoNet_frame_seed
        #         kept_mask[i, j, idx[i, j, len_keep[i, j]:]] = False

        # pred_mask = frame_valid_mask&(~kept_mask)

        return masked_x, pred_mask

    def forward_pretrain_separate(self, data):

        ## 1. MRM
        polygon_pos, polypon_prop, polygon_mask, polygon_key_padding = self.extract_map_feature(data)
        lane_embedding, lane_pos_emb = self.map_encoder(polygon_pos, polypon_prop, polygon_mask)
        
        (
            lane_embedding_masked,
            lane_ids_keep_list,
        ) = self.lane_random_masking(
            lane_embedding, self.lane_mask_ratio, polygon_key_padding
        )

        # transformer stack
        x = lane_embedding_masked + lane_pos_emb
        for blk in self.SpaNet:
            x = blk(x, key_padding_mask=polygon_key_padding)
        x = self.norm_spa(x)
  
        # lane pred loss
        lane_pred_mask = polygon_mask.clone().detach() # attention
        for i, idx in enumerate(lane_ids_keep_list):
            lane_pred_mask[i, idx] = False

        lane_pred = rearrange(self.lane_pred(x), 'b n (p c) -> b n p c', c=2)
        # lane_reg_mask = ~polygon_mask
        # lane_reg_mask[~lane_pred_mask] = False
        polygon_pos_rel = rearrange(polygon_pos, 'b n (p c) -> b n p c', c=2) - polygon_pos[:, :, None, 0:2]
        lane_pred_loss = F.smooth_l1_loss(
            lane_pred[lane_pred_mask], polygon_pos_rel[lane_pred_mask], reduction='mean'
        )
        # 

        ## 2. MTM

        agent_features, agent_category, frame_valid_mask, agent_key_padding, ego_state = self.extract_agent_feature(data, include_future=True)

        agent_embedding = self.agent_projector(agent_features)+self.agent_type_emb(agent_category)[:,:,None,:] # B A D
        (agent_masked_tokens, frame_pred_mask) = self.trajectory_random_masking(agent_embedding, self.trajectory_mask_ratio, frame_valid_mask)

        agent_masked_tokens_ = rearrange(agent_masked_tokens, 'b a t d -> (b a) t d').clone()
        agent_masked_tokens_pos_embeded = self.pe(agent_masked_tokens_)
        agent_tempo_key_padding = rearrange(~frame_valid_mask, 'b a t -> (b a) t') 
        # y, _ = self.tempo_net(agent_masked_tokens_pos_embeded, agent_tempo_key_padding) 
        y, _ = self.tempo_net(agent_masked_tokens_pos_embeded)

        y[y.isnan()] = 0.0

        # y = self.tempo_net(agent_masked_tokens_pos_embeded) # if key_padding_mask should be used here? this causes nan values in loss and needs investigation
        # frame pred loss
        frame_pred = rearrange(self.agent_frame_predictor(y), '(b a) t c -> b a t c', b=agent_features.shape[0], a=agent_features.shape[1])
        agent_pred_loss = F.smooth_l1_loss(
            frame_pred[frame_pred_mask], agent_features[frame_pred_mask][..., :4]
        )

        # 3. TP
        bs, A = data["agent"]["heading"].shape[0:2]

        agent_embedding_tp = rearrange(agent_embedding[:,:,:self.history_steps].clone(), 'b a t d -> (b a) t d')
        agent_tempo_key_padding_tp = agent_tempo_key_padding[:,:self.history_steps].clone()
        # agent_embedding_tp, agent_pos_emb = self.tempo_net(agent_embedding_tp, agent_tempo_key_padding_tp) # if key_padding_mask should be used here? this causes nan values in loss and needs investigation
        agent_embedding_tp, agent_pos_emb = self.tempo_net(agent_embedding_tp)
        
        agent_embedding_tp = rearrange(agent_embedding_tp, '(b a) t c -> b a t c', b=bs, a=A)
        agent_embedding_tp = reduce(agent_embedding_tp, 'b a t c -> b a c', 'max')
        agent_embedding_tp = agent_embedding_tp + rearrange(agent_pos_emb, '(b a) c -> b a c', b=bs, a=A)

        lane_embedding_tp = lane_embedding + lane_pos_emb
        concat = torch.cat([agent_embedding_tp, lane_embedding_tp], dim=1)
        mask_concat = torch.cat([agent_key_padding, polygon_key_padding], dim=1)

        # substitute all NAN values with 0, to prevent the output from containing NAN values
        # it is probably because the layer norm do not handle NAN values well
        concat[concat.isnan()] = 0.0

        x = concat
        for blk in self.SpaNet:
            x = blk(x, key_padding_mask=mask_concat)
        x = self.norm_spa(x)

        # if NaNs have not been removed, using debug console we can see following lines 
        # concat[~mask_concat].isnan().any()
        # tensor(False, device='cuda:0')
        # x[~mask_concat].isnan().any()
        # tensor(True, device='cuda:0')


        tail_prediction = self.agent_tail_predictor(x[:, :A]).view(bs, -1, self.future_steps-1, 4)
        tail_mask = torch.ones_like(frame_valid_mask[:, :, self.history_steps:], dtype=torch.bool)
        tail_pred_mask = tail_mask & frame_valid_mask[:, :, self.history_steps:]
        tail_pred_loss = F.smooth_l1_loss(
            tail_prediction[tail_pred_mask], agent_features[:, :, self.history_steps:][tail_pred_mask][..., :4]
        ) # !!! for several runs, reports "illegal memory access" after several epochs of training

        out = {
            "MRM_loss": lane_pred_loss,
            "MTM_loss": agent_pred_loss,
            "TP_loss": tail_pred_loss,
            "loss": lane_pred_loss + agent_pred_loss + tail_pred_loss,
        }

        assert not torch.isnan(lane_pred_loss).any()
        assert not torch.isnan(agent_pred_loss).any()
        assert not torch.isnan(tail_pred_loss).any()

        return out

    def plot_lane_intention(self, data, lane_intention_score, output_trajectory, key_padding_mask, agent_score=None, k=0):
        i = 0
        polygon_pos, _, polypon_prop, polygon_mask, polygon_key_padding = self.extract_map_feature(data)
        agent_features, agent_category, frame_valid_mask, agent_key_padding, ego_state = self.extract_agent_feature(data, include_future=False)
        # assert agent_features.shape[1]+polygon_pos.shape[1] == attn_weights.shape[1]
        assert polygon_pos.shape[1] == lane_intention_score.shape[1]
        map_points = polygon_pos[i]
        plot_scene_attention(agent_features[i], frame_valid_mask[i], map_points, lane_intention_score[i],
                             key_padding_mask[i, :], 
                              output_trajectory[i], agent_score[i], filename=self.inference_counter, prefix=k)
        

    # def attention_guided_mask_generation(self, attn_weights, key_padding_mask):
    #     '''
    #     This function generates N_mask antagonistic masks. 
    #     All the masks are generated based on the attention weights of the last layer of the transformer encoder.
    #     All the masks belong to one set should sum up to the key_padding_mask corresponding to that scene. 
    #     All the masks should contain roughly the same number of preserved values.

    #     attn_weights: (B, 1+num_seed+M)
    #     key_padding_mask: (B, M)
    #     '''
    #     attn_weights = attn_weights[:, (1+self.rep_seeds_num):] # (B, M)
    #     key_padding_mask = key_padding_mask[:, (1+self.rep_seeds_num):]
    #     # firstly, sort the attention weights
    #     sorted_idx = torch.argsort(attn_weights, dim=-1) # (B, M)
    #     # randint = torch.randint(0, self.N_mask, key_padding_mask.shape, device=key_padding_mask.device)
    #     randint = self.randint[:key_padding_mask.shape[0], :key_padding_mask.shape[1]]
    #     # now generate the masks shaped (B, N_mask, M)
    #     masks = torch.zeros((self.N_mask, *key_padding_mask.shape), device=key_padding_mask.device, dtype=torch.bool)
    #     for i in range(self.N_mask):
    #         mask = torch.zeros_like(key_padding_mask)
    #         for j in range(key_padding_mask.shape[0]):
    #             mask[j, sorted_idx[j, randint[j]==i]] = True
    #         # element being 1 in mask should be kept
    #         masks[i] = key_padding_mask | (~mask)

    #     assert ((~masks).sum(0) == ~key_padding_mask).all()
        
    #     return masks
    def generate_antagonistic_masks(self, key_padding_mask, random_ratio, score=None):
        '''
        This function generates 2 antagonistic masks. 
        All the masks belong to one set should sum up to the key_padding_mask corresponding to that scene. 
        Each of the mask in one set contains roughly the same number of preserved values.

        key_padding_mask: (B, N, M)
        '''

        # if random_ratio == 1.0:
        #     rand_score = torch.rand(key_padding_mask.shape, device=key_padding_mask.device) # range: [0, 1)
        #     rand_score[key_padding_mask] = -2 # later when sorting, the masked values will be put at the end
        # else:
        #     assert score.shape == key_padding_mask.shape
        #     score[key_padding_mask] = -2 # score's range is (-1, 1). later when sorting, the masked values will be put at the end

        # firstly, generate a mask with random_ratio preserved values randomly

        rand_score = torch.rand(key_padding_mask.shape, device=key_padding_mask.device) # range: [0, 1)
        rand_score[key_padding_mask] = -2 # later when sorting, the masked values will be put at the end

        valid_num = (~key_padding_mask).sum(-1)
        random_elem_num = torch.floor(valid_num * random_ratio * 0.5).to(torch.int)
        score_based_elem_num = torch.floor(valid_num * 0.5).to(torch.int) - random_elem_num
        
        randidx = torch.argsort(rand_score, dim=-1, descending=True) # from high score to low score

        random_mask = torch.ones((key_padding_mask.shape[0], key_padding_mask.shape[1]), device=key_padding_mask.device, dtype=torch.bool)

        for i in range(key_padding_mask.shape[0]):
            random_mask[i, randidx[i, :random_elem_num[i]]] = False

        remaining_valid_padding_mask = ~((~key_padding_mask) & random_mask)
        score[remaining_valid_padding_mask] = -2 # score's range is (-1, 1). later when sorting, the masked values will be put at the end
        score_ranked_idx = torch.argsort(score, dim=-1, descending=True) # from high score to low score

        score_mask = torch.ones((key_padding_mask.shape[0], key_padding_mask.shape[1]), device=key_padding_mask.device, dtype=torch.bool)
        for i in range(key_padding_mask.shape[0]):
            score_mask[i, score_ranked_idx[i, :score_based_elem_num[i]]] = False

        mask_0 = random_mask & score_mask
        mask_1 = ~((~key_padding_mask) & mask_0)

        mask_pair = torch.stack([mask_0, mask_1], dim=1)
            
        # masks[:,:,0] = False # keep the ego query always
        assert ((~mask_pair).sum(1) == ~key_padding_mask).all() # all sum up to 1

        # the following part will cause CUDA assert error and is not necessary in principle
        # if not (~masks).sum(-1).all():
        #     print('There exists a mask that does not contain any preserved value')
        #     problem_pos = torch.nonzero((~masks).sum(-1)==0)
        #     for i in problem_pos.shape[0]:
        #         masks[problem_pos[i]] = key_padding_mask_sliced[problem_pos[i,-1]] # keep the original key_padding_mask
        #     print('Successfully dealt')
        
        return mask_pair

    def forward_CME_pretrain(self, data):
        bs, A = data["agent"]["heading"].shape[0:2]
        # x_orig, key_padding_mask = self.embed(data, torch.cat((self.plan_seed, self.rep_seed)))
        x_orig, key_padding_mask, route_key_padding_mask, lane_intention_2s_gt, lane_intention_8s_gt, waypoints_gt = self.embed(data, training=True)
        x_agent = x_orig[:, :A]
        x_map = x_orig[:, A:]
        h_agent = torch.mean(x_agent*(~key_padding_mask[:,:A,None]), dim=1)
        h_map = torch.mean(x_map*(~key_padding_mask[:,A:,None]), dim=1)

        z_motion = self.cme_motion_mlp(h_agent)
        z_env = self.cme_env_mlp(h_map)

        v_loss_motion = self.variance_loss(z_motion)
        v_loss_env = self.variance_loss(z_env)
        c_loss_motion = self.covariance_loss(z_motion)
        c_loss_env = self.covariance_loss(z_env)
        inv_loss = self.invariance_loss(z_motion, z_env)

        v_loss = v_loss_motion + v_loss_env
        c_loss = c_loss_motion + c_loss_env

        out = {
            "loss": 10*v_loss + 0.5*c_loss + 2.5*inv_loss,
            "v_loss": v_loss,
            "c_loss": c_loss,
            "inv_loss": inv_loss,
        }

        return out
    

    def forward_teacher_enforcing(self, data):
        bs, A = data["agent"]["heading"].shape[0:2]
        # x_orig, key_padding_mask = self.embed(data, torch.cat((self.plan_seed, self.rep_seed)))
        x_orig, key_padding_mask, route_key_padding_mask, lane_intention_2s_gt, lane_intention_8s_gt, waypoints_gt = self.embed(data, training=True)

        # no need to remove the ego token here. Right?

        x = x_orig 
        for blk in self.SpaNet:
            x = blk(x, key_padding_mask=key_padding_mask)
        x = self.norm_spa(x)

        abs_prediction = self.abs_agent_predictor(x[:, 1:A]).view(bs, -1, self.future_steps, 2)
        lane_intention_2s = self.lane_intention_2s_predictor(x[:, A:]).squeeze(-1) # B M
        lane_intention_2s[key_padding_mask[:, A:]] = -torch.inf # except the route, all the other map elements are set to -inf # lane_intention[key_padding_mask[:, A:]] = -torch.inf # invalide map elements are set to -inf. non-route lane segments remain possible to be selected, # since after disable route correction, the density of route lane segments gets lower
        lane_intention_2s_prob = F.softmax(lane_intention_2s, dim=-1)

        lane_intention_8s = self.lane_intention_8s_predictor(x[:, A:]).squeeze(-1) # B M
        lane_intention_8s[key_padding_mask[:, A:]] = -torch.inf # except the route, all the other map elements are set to -inf
        lane_intention_8s_prob = F.softmax(lane_intention_8s, dim=-1)

        loss_lane_intention_2s = F.cross_entropy(lane_intention_2s_prob, lane_intention_2s_gt, reduction="none")
        loss_lane_intention_8s = F.cross_entropy(lane_intention_8s_prob, lane_intention_8s_gt, reduction="none")

        # find the lane segment with the highest probability
        lane_intention_max_2s = lane_intention_2s_prob.argmax(dim=-1)
        lane_intention_correct_rates_2s = (lane_intention_max_2s == lane_intention_2s_gt).float().mean()
        lane_intention_topk_2s = lane_intention_2s_prob.topk(k=6, dim=-1, largest=True, sorted=False).indices
        lane_intention_topk_correct_rate_2s = (lane_intention_topk_2s == lane_intention_2s_gt.unsqueeze(-1)).any(-1).float().mean()

        lane_intention_max_8s = lane_intention_8s_prob.argmax(dim=-1)
        lane_intention_correct_rates_8s = (lane_intention_max_8s == lane_intention_8s_gt).float().mean()
        lane_intention_topk_8s = lane_intention_8s_prob.topk(k=6, dim=-1, largest=True, sorted=False).indices
        lane_intention_topk_correct_rate_8s = (lane_intention_topk_8s == lane_intention_8s_gt.unsqueeze(-1)).any(-1).float().mean()

        # intention_lane_seg = x[:, A:][torch.arange(bs), lane_intention_max]
        # assert route_key_padding_mask[torch.arange(bs), lane_intention_max].any() == False # assert the selected lane segment is on the route

        intention_lane_seg_2s = self.seed_2s_mlp(x_orig[:, A:][torch.arange(bs), lane_intention_2s_gt].detach().clone())
        intention_lane_seg_8s = self.seed_8s_mlp(x_orig[:, A:][torch.arange(bs), lane_intention_8s_gt].detach().clone())
        # assert route_key_padding_mask[torch.arange(bs), lane_intention_targets].any() == False # assert the selected lane segment is on the route
        # The above assertion would cause error, probably because there are scenarios where no route lane is known

        ################ WpNet ################ 
        # attraction_point = attraction_point_gt
        # q = (self.attraction_point_projector(attraction_point)+self.lane_emb_cr_mlp(intention_lane_seg)).unsqueeze(1)
        x_wpnet = torch.cat([self.lane_emb_wp_2s_mlp(intention_lane_seg_2s).unsqueeze(1),
                            #  self.lane_emb_wp_8s_mlp(intention_lane_seg_8s).unsqueeze(1),
                              x_orig[:,1:]], dim=1)
        key_padding_mask_wp = torch.cat([torch.zeros((bs, 1), dtype=torch.bool, device=key_padding_mask.device),
                                          key_padding_mask[:, 1:]], dim=1)
        for blk in self.WpNet:
            x_wpnet = blk(x_wpnet, key_padding_mask=key_padding_mask_wp)
        x_wpnet = self.norm_wp(x_wpnet)
        rel_prediction = self.rel_agent_predictor(x_wpnet[:, 1:A]).view(bs, -1, self.waypoints_number, 2)
        waypoints = self.waypoint_decoder(x_wpnet[:, 0]) # B T_wp 4

        ################ FFNet ################
        x_ffnet = torch.cat([
            # self.lane_emb_ff_2s_mlp(intention_lane_seg_2s).unsqueeze(1),
                            self.lane_emb_ff_8s_mlp(intention_lane_seg_8s).unsqueeze(1),
                            self.waypoints_embedder(waypoints_gt.view(bs, -1)).unsqueeze(1),
                            x_orig], dim=1)
        
        key_padding_mask_ff = torch.cat([torch.zeros((bs, 2), dtype=torch.bool, device=key_padding_mask.device),
                                          key_padding_mask], dim=1)
        for blk in self.FFNet:
            x_ffnet = blk(x_ffnet, key_padding_mask=key_padding_mask_ff)
        x_ffnet = self.norm_ff(x_ffnet)
        far_future_traj = self.far_future_traj_decoder(x_ffnet[:, 0])

        # decode the far-future trajectory
        # attraction_point = waypoints[:, -1] 

        # restrict the attention to the route only in the cross attender
        # q, attn_weights = blk(query=q, key=x, value=x, key_padding_mask=route_kpmask, need_weights=True) 

        trajectory = torch.cat([waypoints, far_future_traj], dim=1).unsqueeze(1) # B 1 T 4
        probability = torch.ones(bs, 1, device=trajectory.device) # B 1 

        assert trajectory.isnan().any() == False

        out = {
            "trajectory": trajectory,
            "probability": probability,
            "prediction": abs_prediction,
            "rel_prediction" : rel_prediction,
            "waypoints": waypoints,
            "far_future_traj": far_future_traj,
            "lane_intention_loss": loss_lane_intention_2s+loss_lane_intention_8s,
            "lane_intention_correct_rates": lane_intention_correct_rates_2s,
            "lane_intention_topk_correct_rate": lane_intention_topk_correct_rate_2s,
            "lane_intention_correct_rates_8s": lane_intention_correct_rates_8s,
            "lane_intention_topk_correct_rate_8s": lane_intention_topk_correct_rate_8s,
        }

        if not self.training:
            output_trajectory = trajectory[:, 0]
            angle = torch.atan2(output_trajectory[..., 3], output_trajectory[..., 2])
            out["output_trajectory"] = torch.cat(
                [output_trajectory[..., :2], angle.unsqueeze(-1)], dim=-1
            )

        # attention visualization
        if False:
            attn_weights = self.SpaNet[-1].attn_mat[:, 0].detach()
            # visualize the scene using the attention weights
            self.plot_lane_intention(data, attn_weights, output_trajectory, key_padding_mask, 0)
            self.inference_counter += 1

        return out

    def forward_multimodal_finetune(self, data):
        bs, A = data["agent"]["heading"].shape[0:2]
        # x_orig, key_padding_mask = self.embed(data, torch.cat((self.plan_seed, self.rep_seed)))
        x_orig, key_padding_mask, route_key_padding_mask = self.embed(data, training=False)

        # no need to remove the ego token here. Right?

        x = x_orig 
        for blk in self.SpaNet:
            x = blk(x, key_padding_mask=key_padding_mask)
        x = self.norm_spa(x)

        abs_prediction = self.abs_agent_predictor(x[:, 1:A]).view(bs, -1, self.future_steps, 2)
        lane_intention_2s = self.lane_intention_2s_predictor(x[:, A:]).squeeze(-1) # B M
        lane_intention_2s[key_padding_mask[:, A:]] = -torch.inf # except the route, all the other map elements are set to -inf # lane_intention[key_padding_mask[:, A:]] = -torch.inf # invalide map elements are set to -inf. non-route lane segments remain possible to be selected, # since after disable route correction, the density of route lane segments gets lower

        lane_intention_8s = self.lane_intention_8s_predictor(x[:, A:]).squeeze(-1) # B M
        lane_intention_8s[key_padding_mask[:, A:]] = -torch.inf # except the route, all the other map elements are set to -inf

        lane_intention_2s_prob = F.softmax(lane_intention_2s, dim=-1)
        lane_intention_8s_prob = F.softmax(lane_intention_8s, dim=-1)

        lane_intention_topk_2s = lane_intention_2s_prob.topk(k=self.num_modes, dim=-1, largest=True, sorted=False).indices # [B, M]
        lane_intention_topk_8s = lane_intention_8s_prob.topk(k=self.num_modes, dim=-1, largest=True, sorted=False).indices

        waypoints_list = []
        far_future_traj_list = []
        rel_prediction_list = []

        # for i in range(self.num_modes):
        for i in range(self.num_modes):
            intention_lane_seg_2s = self.seed_2s_mlp(x_orig[:, A:][torch.arange(bs), lane_intention_topk_2s[:, i]].detach().clone())

            x_wpnet = torch.cat([self.lane_emb_wp_2s_mlp(intention_lane_seg_2s).unsqueeze(1),
                            #  self.lane_emb_wp_8s_mlp(intention_lane_seg_8s).unsqueeze(1),
                              x_orig[:,1:]], dim=1)
            key_padding_mask_wp = torch.cat([torch.zeros((bs, 1), dtype=torch.bool, device=key_padding_mask.device),
                                            key_padding_mask[:, 1:]], dim=1)
            for blk in self.WpNet:
                x_wpnet = blk(x_wpnet, key_padding_mask=key_padding_mask_wp)
            x_wpnet = self.norm_wp(x_wpnet)
            rel_prediction = self.rel_agent_predictor(x_wpnet[:, 1:A]).view(bs, -1, self.waypoints_number, 2)
            waypoints = self.waypoint_decoder(x_wpnet[:, 0]) # B T_wp 4

            waypoints_list.append(waypoints)
            rel_prediction_list.append(rel_prediction)

            ################ FFNet ################
            for k in range(self.num_modes):
                intention_lane_seg_8s = self.seed_8s_mlp(x_orig[:, A:][torch.arange(bs), lane_intention_topk_8s[:, k]].detach().clone())

                x_ffnet = torch.cat([
                # self.lane_emb_ff_2s_mlp(intention_lane_seg_2s).unsqueeze(1),
                            self.lane_emb_ff_8s_mlp(intention_lane_seg_8s).unsqueeze(1),
                            self.waypoints_embedder(waypoints.detach().clone().view(bs, -1)).unsqueeze(1),
                            x_orig], dim=1)
        
                key_padding_mask_ff = torch.cat([torch.zeros((bs, 2), dtype=torch.bool, device=key_padding_mask.device),
                                                key_padding_mask], dim=1)
                for blk in self.FFNet:
                    x_ffnet = blk(x_ffnet, key_padding_mask=key_padding_mask_ff)
                x_ffnet = self.norm_ff(x_ffnet)
                far_future_traj = self.far_future_traj_decoder(x_ffnet[:, 0])

                far_future_traj_list.append(far_future_traj)


        multimodal_waypoints = torch.stack(waypoints_list, dim=1) # B M T_wp 4
        multimodal_rel_prediction = torch.stack(rel_prediction_list, dim=1) # B M A T_wp 2

        multimodal_far_future_traj = torch.stack(far_future_traj_list, dim=1)
        fshape = multimodal_far_future_traj.shape
        multimodal_far_future_traj = multimodal_far_future_traj.reshape(fshape[0], self.num_modes, fshape[1]//self.num_modes, fshape[2], fshape[3])

        trajectory = torch.cat([multimodal_waypoints, multimodal_far_future_traj[:, :, 0]], dim=2) # B M T 4
        probability = torch.zeros(bs, self.num_modes, device=trajectory.device) # B M

        probability[:, 0] = 1.0

        out = {
            "trajectory": trajectory,
            "probability": probability,
            "prediction": abs_prediction,
            "rel_prediction" : multimodal_rel_prediction,
            "waypoints": multimodal_waypoints,
            "far_future_traj": multimodal_far_future_traj,
            "lane_intention_2s_prob": lane_intention_2s_prob,
            "lane_intention_8s_prob": lane_intention_8s_prob,
            "lane_intention_topk_2s": lane_intention_topk_2s,
            "lane_intention_topk_8s": lane_intention_topk_8s,
        }

        if not self.training:
            output_trajectory = trajectory[:, 0]
            angle = torch.atan2(output_trajectory[..., 3], output_trajectory[..., 2])
            out["output_trajectory"] = torch.cat(
                [output_trajectory[..., :2], angle.unsqueeze(-1)], dim=-1
            )

        # attention visualization
        if False:
            ## visualize WPNet
            # attn_weights = self.WpNet[-1].attn_mat[:, 0].detach()
            # # score_to_visualize = lane_intention_prob_2s
            # score_to_visualize = attn_weights[:, 1+A:]
            # # visualize the scene using the attention weights
            # self.plot_lane_intention(data, score_to_visualize, output_trajectory, key_padding_mask, attn_weights[:, 1:1+A], 0)
            # self.inference_counter += 1

            ## visualize FFNet
            attn_weights = self.FFNet[-1].attn_mat[:, 0].detach()
            # score_to_visualize = lane_intention_prob_2s
            score_to_visualize = attn_weights[:, 3+A:]
            # visualize the scene using the attention weights
            self.plot_lane_intention(data, score_to_visualize, output_trajectory, key_padding_mask, attn_weights[:, 3:3+A], 0)
            self.inference_counter += 1

        return out
    
    def forward_inference(self, data):
        bs, A = data["agent"]["heading"].shape[0:2]
        # x_orig, key_padding_mask = self.embed(data, torch.cat((self.plan_seed, self.rep_seed)))
        x_orig, key_padding_mask, route_key_padding_mask = self.embed(data, training=False)

        # no need to remove the ego token here. Right?

        x = x_orig 
        for blk in self.SpaNet:
            x = blk(x, key_padding_mask=key_padding_mask)
        x = self.norm_spa(x)

        abs_prediction = self.abs_agent_predictor(x[:, 1:A]).view(bs, -1, self.future_steps, 2)
        lane_intention_2s = self.lane_intention_2s_predictor(x[:, A:]).squeeze(-1) # B M
        lane_intention_2s[key_padding_mask[:, A:]] = -torch.inf # except the route, all the other map elements are set to -inf # lane_intention[key_padding_mask[:, A:]] = -torch.inf # invalide map elements are set to -inf. non-route lane segments remain possible to be selected, # since after disable route correction, the density of route lane segments gets lower

        lane_intention_8s = self.lane_intention_8s_predictor(x[:, A:]).squeeze(-1) # B M
        lane_intention_8s[key_padding_mask[:, A:]] = -torch.inf # except the route, all the other map elements are set to -inf

        lane_intention_prob_2s = F.softmax(lane_intention_2s, dim=-1)
        lane_intention_prob_8s = F.softmax(lane_intention_8s, dim=-1)

        # find the lane segment with the highest probability
        lane_intention_max_2s = lane_intention_prob_2s.argmax(dim=-1)
        lane_intention_max_8s = lane_intention_prob_8s.argmax(dim=-1)

        # Ensure indices are within bounds
        # assert lane_intention_max.max() < x[:, A:].shape[1], "Index out of bounds in lane_intention_max"

        intention_lane_seg_2s = self.seed_2s_mlp(x_orig[:, A:][torch.arange(bs), lane_intention_max_2s].detach().clone())
        intention_lane_seg_8s = self.seed_8s_mlp(x_orig[:, A:][torch.arange(bs), lane_intention_max_8s].detach().clone())
        # assert route_key_padding_mask[torch.arange(bs), lane_intention_max].any() == False # assert the selected lane segment is on the route

        ################ WpNet ################ 
        # attraction_point = attraction_point_gt
        # q = (self.attraction_point_projector(attraction_point)+self.lane_emb_cr_mlp(intention_lane_seg)).unsqueeze(1)
        x_wpnet = torch.cat([self.lane_emb_wp_2s_mlp(intention_lane_seg_2s).unsqueeze(1),
                            #  self.lane_emb_wp_8s_mlp(intention_lane_seg_8s).unsqueeze(1),
                              x_orig[:,1:]], dim=1)
        key_padding_mask_wp = torch.cat([torch.zeros((bs, 1), dtype=torch.bool, device=key_padding_mask.device),
                                          key_padding_mask[:, 1:]], dim=1)
        for blk in self.WpNet:
            x_wpnet = blk(x_wpnet, key_padding_mask=key_padding_mask_wp)
        x_wpnet = self.norm_wp(x_wpnet)
        rel_prediction = self.rel_agent_predictor(x_wpnet[:, 1:A]).view(bs, -1, self.waypoints_number, 2)
        waypoints = self.waypoint_decoder(x_wpnet[:, 0]) # B T_wp 4

        ################ FFNet ################
        x_ffnet = torch.cat([
            # self.lane_emb_ff_2s_mlp(intention_lane_seg_2s).unsqueeze(1),
                            self.lane_emb_ff_8s_mlp(intention_lane_seg_8s).unsqueeze(1),
                            self.waypoints_embedder(waypoints.view(bs, -1).detach().clone()).unsqueeze(1),
                            x_orig], dim=1)
        
        key_padding_mask_ff = torch.cat([torch.zeros((bs, 2), dtype=torch.bool, device=key_padding_mask.device),
                                          key_padding_mask], dim=1)
        for blk in self.FFNet:
            x_ffnet = blk(x_ffnet, key_padding_mask=key_padding_mask_ff)
        x_ffnet = self.norm_ff(x_ffnet)
        far_future_traj = self.far_future_traj_decoder(x_ffnet[:, 0])

        trajectory = torch.cat([waypoints, far_future_traj], dim=1).unsqueeze(1) # B 1 T 4
        probability = torch.ones(bs, 1, device=trajectory.device) # B 1 

        out = {
            "trajectory": trajectory,
            "probability": probability,
            "prediction": abs_prediction,
            "rel_prediction" : rel_prediction,
            "waypoints": waypoints,
            "far_future_traj": far_future_traj,
        }

        if not self.training:
            output_trajectory = trajectory[:, 0]
            angle = torch.atan2(output_trajectory[..., 3], output_trajectory[..., 2])
            out["output_trajectory"] = torch.cat(
                [output_trajectory[..., :2], angle.unsqueeze(-1)], dim=-1
            )

        # attention visualization
        if False:
            ## visualize WPNet
            # attn_weights = self.WpNet[-1].attn_mat[:, 0].detach()
            # # score_to_visualize = lane_intention_prob_2s
            # score_to_visualize = attn_weights[:, 1+A:]
            # # visualize the scene using the attention weights
            # self.plot_lane_intention(data, score_to_visualize, output_trajectory, key_padding_mask, attn_weights[:, 1:1+A], 0)
            # self.inference_counter += 1

            ## visualize FFNet
            attn_weights = self.FFNet[-1].attn_mat[:, 0].detach()
            # score_to_visualize = lane_intention_prob_2s
            score_to_visualize = attn_weights[:, 3+A:]
            # visualize the scene using the attention weights
            self.plot_lane_intention(data, score_to_visualize, output_trajectory, key_padding_mask, attn_weights[:, 3:3+A], 0)
            self.inference_counter += 1

        return out


    def forward_antagonistic_mask_finetune(self, data, current_epoch=None):

        if current_epoch is None: 
            random_ratio = 0.0
        else:
            random_ratio = 1-(current_epoch-self.pretrain_epoch_stages[2])/10
            random_ratio = max(random_ratio, 0.2)
        

        bs, A = data["agent"]["heading"].shape[0:2]
        # x_orig, key_padding_mask = self.embed(data, torch.cat((self.plan_seed, self.rep_seed)))
        x_orig, key_padding_mask = self.embed(data, need_route_kpmask=False)

        x = x_orig 
        for blk in self.SpaNet:
            x = blk(x, key_padding_mask=key_padding_mask)
        x = self.norm_spa(x)
        prediction = self.abs_agent_predictor(x[:, 1:A+1]).view(bs, -1, self.future_steps, 2)
        score = self.score_mlp(x[:, 1:]).squeeze(-1)

        masks_3d = self.generate_antagonistic_masks(key_padding_mask[:, 1:], random_ratio, score.detach().clone()) # B, N_mask, M
        masks = rearrange(masks_3d, 'b n m -> (b n) m')
        
        x_m = repeat(x[:, 1:], 'b m d -> (b n) m d', n=self.N_mask)
        q = repeat(x[:, 0:1], 'b m d -> (b n) m d', n=self.N_mask)

        q, attn_weights = self.FFNet[0](query=q, key=x_m, value=x_m, key_padding_mask=masks, need_weights=True)
        goal = self.goal_mlp(q).view(bs, 2, self.out_channels)
        q, attn_weights = self.FFNet[1](query=q, key=x_m, value=x_m, key_padding_mask=masks, need_weights=True)
        waypoints= self.waypoints_mlp(q).view(bs, 2, self.future_steps//self.waypoints_number, self.out_channels)
        q, attn_weights = self.FFNet[2](query=q, key=x_m, value=x_m, key_padding_mask=masks, need_weights=True)
        # restrict the attention to the route only in the cross attender
        # q, attn_weights = blk(query=q, key=x, value=x, key_padding_mask=route_kpmask, need_weights=True) 

        trajectory, probability = self.trajectory_decoder(q)
        
        trajectory = rearrange(trajectory, '(b n) m t c -> b n m t c', n=self.N_mask)
        probability = rearrange(probability, '(b n) m -> b n m', n=self.N_mask)

        assert trajectory.isnan().any() == False
        assert probability.isnan().any() == False
        assert prediction.isnan().any() == False

        out = {
            "trajectory": trajectory,
            "probability": probability,
            "prediction": prediction,
            "goal": goal,
            "waypoints": waypoints,
            "score": score,
            "masks": masks_3d
        }

        if not self.training:
            probability = probability[:, 0]
            trajectory = trajectory[:, 0]
            out = {
                "trajectory": trajectory,
                "probability": probability,
                "prediction": prediction,
                "goal": goal[:, 0],
                "waypoints": waypoints[:, 0],
            }

            best_mode = probability.argmax(dim=-1)
            output_trajectory = trajectory[torch.arange(bs), best_mode]
            angle = torch.atan2(output_trajectory[..., 3], output_trajectory[..., 2])
            out["output_trajectory"] = torch.cat(
                [output_trajectory[..., :2], angle.unsqueeze(-1)], dim=-1
            )



        # score visualization
        if True:
            # attn_weights = self.SpaNet[-1].attn_mat[:, 0].detach()
            # visualize the scene using the attention weights
            assert bs == 1
            sorted_score, sorted_idx = torch.sort(score[0], descending=True)
            score[0, sorted_idx[sorted_idx.shape[0]//2:]] = 0
            self.plot_lane_intention(data, score, output_trajectory, key_padding_mask, 0)
            self.inference_counter += 1

        return out
    
    
    def embed(self, data, training=False):
        """
            Extract the features of the map and the agents, and then embed them into the latent space. 
            Instead of using the embedding of ego history trajectory, we use ego velocity token. 

            data: dict
            seeds: tensor (D, )

            returns: 
                x: tensor (B, N, D)
                key_padding_mask: tensor (B, N)
                route_kp_mask: tensor (B, M)
            if training is True, then also returns:
                lane_intention_targets: tensor (B, M)
                rel_targets: tensor (B, A, T, 2)

        """
        
        polygon_pos, polygon_ori, polypon_prop, polygon_mask, polygon_key_padding, route_kp_mask = self.extract_map_feature(data, need_route_kpmask=True)
    
        # lane embedding
        lane_embedding, lane_pos_emb = self.map_encoder(polygon_pos, polypon_prop, polygon_mask)
        lane_embedding_pos = lane_embedding + lane_pos_emb

        # agent embedding (not dropping frames out )
        if training:    
            agent_features, agent_category, frame_valid_mask, agent_key_padding, ego_state,\
                attraction_point, horizon_point, waypoints_gt = self.extract_agent_feature(data, include_future=False, get_critical_points=True)
        else:
            agent_features, agent_category, frame_valid_mask, agent_key_padding, ego_state = self.extract_agent_feature(data, include_future=False)

        # ego state embedding (currently only velocity)
        ego_vel_token = self.vel_token_projector(ego_state[:, 3:4]).unsqueeze(1) # B 1 D

        bs, A = agent_features.shape[0:2]
        agent_embedding = self.agent_projector(agent_features)+self.agent_type_emb(agent_category)[:,:,None,:] # B A D

        # # if agent frames should be masked here?
        # 1. if not, use following code

        # (agent_masked_tokens, frame_pred_mask) = self.trajectory_random_masking(agent_embedding, self.trajectory_mask_ratio, frame_valid_mask)
        agent_embedding[~frame_valid_mask] = self.TempoNet_frame_seed
        agent_embedding_ft = rearrange(agent_embedding.clone(), 'b a t d -> (b a) t d')
        # agent_tempo_key_padding = rearrange(~frame_valid_mask, 'b a t -> (b a) t')
        
        agent_embedding_ft = self.pe(agent_embedding_ft)
        # agent_embedding_ft, agent_pos_emb = self.tempo_net(agent_embedding_ft, agent_tempo_key_padding) # if key_padding_mask should be used here? this causes nan values in loss and needs investigation
        agent_embedding_emb, agent_pos_emb = self.tempo_net(agent_embedding_ft, agent_features[:, :, -1, :4].view(bs*A, 4))

        # 2. if yes, use following code

        # (agent_masked_tokens, frame_pred_mask) = self.trajectory_random_masking(agent_embedding, self.trajectory_mask_ratio, frame_valid_mask)
        # agent_masked_tokens_ = rearrange(agent_masked_tokens, 'b a t d -> (b a) t d').clone()
        # agent_masked_tokens_pos_embeded = self.pe(agent_masked_tokens_)
        # agent_tempo_key_padding = rearrange(~frame_valid_mask, 'b a t -> (b a) t') 
        # # y, _ = self.tempo_net(agent_masked_tokens_pos_embeded, agent_tempo_key_padding) 
        # agent_embedding_emb, agent_pos_emb = self.tempo_net(agent_masked_tokens_pos_embeded)

        #############

        agent_embedding_emb = rearrange(agent_embedding_emb, '(b a) t c -> b a t c', b=bs, a=A)
        agent_embedding_emb = reduce(agent_embedding_emb, 'b a t c -> b a c', 'max')
        agent_embedding_emb = agent_embedding_emb + rearrange(agent_pos_emb, '(b a) c -> b a c', b=bs, a=A)

        x = torch.cat([ego_vel_token, agent_embedding_emb[:, 1:], lane_embedding_pos], dim=1) # drop the ego history token here
        # key padding masks
        # key_padding_mask = torch.cat([torch.zeros(ego_vel_token.shape[0:2], device=agent_key_padding.device, dtype=torch.bool), agent_key_padding, polygon_key_padding], dim=-1)
        
        key_padding_mask = torch.cat([agent_key_padding, polygon_key_padding], dim=-1)

        res = [x, key_padding_mask, route_kp_mask]

        if training:

            polygon_pos_and_ori = torch.cat([polygon_pos[..., :2], polygon_ori.cos(), polygon_ori.sin()], dim=-1)
            # lane_intention_target 2s
            dist = torch.norm( polygon_pos_and_ori - attraction_point[:, None, None, :], dim=-1) # [B, M, S]
            ori_diff_2s = torch.norm(polygon_pos_and_ori[..., 2:] - attraction_point[:, None, None, 2:], dim=-1) # [B, M]
            diff_above_threshold = ori_diff_2s > self.ori_threshold
            dist[polygon_key_padding] = torch.inf
            dist[diff_above_threshold] = torch.inf
            lane_intention_2s = dist.min(dim=-1)[0].argmin(dim=-1) # B
            # lane_intention_target 8s
            dist = torch.norm(polygon_pos_and_ori - horizon_point[:, None, None, :], dim=-1) # [B, M, S]
            ori_diff_8s = torch.norm(polygon_pos_and_ori[..., 2:] - horizon_point[:, None, None, 2:], dim=-1) # [B, M]
            diff_above_threshold = ori_diff_8s > self.ori_threshold
            dist[polygon_key_padding] = torch.inf
            dist[diff_above_threshold] = torch.inf
            lane_intention_8s = dist.min(dim=-1)[0].argmin(dim=-1) # B

            res.extend([lane_intention_2s, lane_intention_8s, waypoints_gt])
        
        return res


    def mask_and_embed(self, data, seeds): # 
        """
            IMPORTANT: unmaintained, unusable now
            data: dict
            seeds: tensor (D, )
        """

        agent_features, agent_category, frame_valid_mask, agent_key_padding, ego_state = self.extract_agent_feature(data)
        bs, A = agent_features.shape[0:2]
        # agent_key_padding = ~(agent_mask.any(-1))
        if seeds.dim() == 1:
            seeds = repeat(seeds, 'd -> bs 1 d', bs=bs)
        else:
            seeds = repeat(seeds, 'n d -> bs n d', bs=bs) 

        ## lane masking
        map_features, polygon_mask, polygon_key_padding = self.extract_map_feature(data)

        (
            lane_masked_tokens,
            lane_ids_keep_list,
        ) = self.lane_random_masking(
            map_features, self.lane_mask_ratio, polygon_key_padding
        )

        lane_embedding = self.map_encoder(lane_masked_tokens)

        agent_embedding = self.agent_projector(agent_features) # B A D
        (agent_masked_tokens, frame_pred_mask) = self.trajectory_random_masking(agent_embedding, self.trajectory_mask_ratio, frame_valid_mask)

        b, a = agent_masked_tokens.shape[0:2]
        agent_masked_tokens_ = rearrange(agent_masked_tokens, 'b a t d -> (b a) t d').clone()
        agent_masked_tokens_pos_embeded = self.pe(agent_masked_tokens_)
        # agent_tempo_key_padding = rearrange(~agent_mask, 'b a t -> (b a) t')

        x_agent = self.tempo_net(agent_masked_tokens_pos_embeded)
        x_agent_ = rearrange(x_agent, '(b a) t c -> b a t c', b=b, a=a)
        x_agent_ = reduce(x_agent_, 'b a t c -> b a c', 'max')

        x = torch.cat([seeds, x_agent_, lane_embedding], dim=1)

        # key padding masks
        agent_key_padding = ~(frame_valid_mask.any(-1))
        polygon_key_padding = ~(polygon_mask.any(-1))
        key_padding_mask = torch.cat([torch.zeros(seeds.shape[0:2], device=agent_key_padding.device, dtype=torch.bool), agent_key_padding, polygon_key_padding], dim=-1)

        return x, key_padding_mask

    def forward_pretrain_representation(self, data):
        data_distorted = self.distortor.augment(data)
        x_stu, x_stu_key_padding_mask = self.mask_and_embed(data_distorted, self.rep_seed)

        # forward through student model 
        for blk in self.SpaNet:
            x_stu = blk(x_stu, key_padding_mask=x_stu_key_padding_mask)
        x_stu = self.norm_spa(x_stu)

        x, key_padding_mask = self.embed(data, self.rep_seed, include_future=True)
        # rep_seed is concatenated to each beginning of the sequence
        x = x.detach() # the teacher model is not updated by gradient descent
        # forward through teacher model TODO: to modify 
        for blk in self.blocks_teacher:
            x = blk(x, key_padding_mask=key_padding_mask)
        x_tch = self.norm_teacher(x)

        # let's say, the first embedding is the representation we want to pretrain
        z = self.expander(rearrange(x_stu[:, 0:self.rep_seed.shape[0]], 'b n d -> b (n d)')) # B expanded_dim
        z_t = self.expander_teacher(rearrange(x_tch[:, 0:self.rep_seed.shape[0]], 'b n d -> b (n d)')) # B expanded_dim

        # calculate the loss (VICReg)
        # 1. variance
        S = torch.sqrt(torch.var(z, dim=0)+1e-6)
        v_loss = torch.mean(torch.clip(self.gamma-S, min=0))
        # 2. covariance
        delta_z = z - torch.mean(z, dim=0, keepdim=True) # B D
        cov = torch.sum(torch.matmul(rearrange(delta_z, 'b d -> b d 1'), rearrange(delta_z, 'b d -> b 1 d'))/(delta_z.shape[0]-1), dim=0) # D D
        cov_off_diag = cov - torch.diag(torch.diagonal(cov))
        c_loss = torch.sum(torch.pow(cov_off_diag, 2))/self.expanded_dim
        # 3. invariance
        inv_loss = torch.mean(torch.norm(z - z_t, dim=-1))

        out = {
            "loss": 2.5*v_loss + 0.1*c_loss + 2.5*inv_loss,
            "v_loss": v_loss,
            "c_loss": c_loss,
            "inv_loss": inv_loss,
        }

        return out
    
    def initialize_teacher(self): 
        for module_teacher, module_student in zip(self.teacher_list, self.student_list):
            for param_teacher, param_student in zip(module_teacher.parameters(), module_student.parameters()):
                param_teacher.data = param_student.data.clone().detach()

    def EMA_update(self):
        for student, teacher in zip(self.student_list, self.teacher_list):
            for param, param_t in zip(student.parameters(), teacher.parameters()):
                param_t.data = self.alpha*param_t.data + (1-self.alpha)*param.data

    def variance_loss(self, z): 
        S = torch.sqrt(torch.var(z, dim=0)+1e-6)
        v_loss = torch.mean(torch.clip(self.gamma-S, min=0))
        return v_loss
    
    def covariance_loss(self, z):
        delta_z = z - torch.mean(z, dim=0, keepdim=True) # B D
        cov = torch.sum(torch.matmul(rearrange(delta_z, 'b d -> b d 1'), rearrange(delta_z, 'b d -> b 1 d')), dim=0)/(delta_z.shape[0]-1) # D D
        cov_off_diag = cov - torch.diag(torch.diagonal(cov))
        c_loss = torch.sum(torch.pow(cov_off_diag, 2))/self.expanded_dim
        return c_loss
    
    def invariance_loss(self, z, z_t):
        inv_loss = torch.mean(torch.norm(z - z_t, dim=-1))
        return inv_loss