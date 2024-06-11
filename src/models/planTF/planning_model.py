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
from einops import rearrange, reduce, repeat
from .layers.embedding import Projector

from enum import Enum
import math

from .debug_vis import plot_lane_segments, plot_scene_attention
from .info_distortor import InfoDistortor


# no meaning, required by nuplan
trajectory_sampling = TrajectorySampling(num_poses=8, time_horizon=8, interval_length=1)

class Stage(Enum):
    PRETRAIN_SEP = 0
    PRETRAIN_MIX = 1
    PRETRAIN_REPRESENTATION = 2
    FINETUNE = 3
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
        pretrain_epoch_stages = [0, 10, 20],
        # pretrain_epoch_stages = [0, 0, 0],
        lane_split_threshold=20,
        alpha=0.999,
        expanded_dim = 256*8,
        gamma = 1.0, # VICReg standard deviation target 
        rep_seeds_num = 4,
        N_mask = 2,
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
        
        self.polygon_channel = polygon_channel # the number of features for each lane segment besides points coords which we will use

        self.lane_mask_ratio = lane_mask_ratio
        self.trajectory_mask_ratio = trajectory_mask_ratio
        self.pretrain_epoch_stages = pretrain_epoch_stages

        self.no_lane_segment_points = 20
        self.lane_split_threshold = lane_split_threshold
        self.alpha = alpha
        self.gamma = gamma
        self.expanded_dim = expanded_dim
        self.rep_seeds_num = rep_seeds_num
        self.N_mask = N_mask

        # modules begin
        self.pe = PositionalEncoding(dim, dropout=0.1, max_len=1000)
        self.pos_emb = build_mlp(4, [dim] * 2)

        self.tempo_net = TempoNet(
            state_channel=state_channel,
            depth=3,
            num_head=8,
            dim_head=dim,
        )
        

        self.agent_seed = nn.Parameter(torch.randn(dim))
        self.rep_seed = nn.Parameter(torch.randn(rep_seeds_num, dim))
        self.plan_seed = nn.Parameter(torch.randn(1, dim))

        self.agent_projector = Projector(dim=dim, in_channels=8) # NOTE: make consistent to state_channel
        self.map_projector = Projector(dim=dim, in_channels=self.no_lane_segment_points*2+5) # NOTE: make consistent to polygon_channel
        self.lane_pred = build_mlp(dim, [512, self.no_lane_segment_points*2])
        self.agent_frame_pred = build_mlp(dim, [512, 3])

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
        self.expander = nn.Linear(dim*rep_seeds_num, expanded_dim)

        self.blocks_teacher = nn.ModuleList(
            Block(
                dim=dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop_path=dpr[i],
            )
            for i in range(encoder_depth)
        )
        self.norm_teacher = nn.LayerNorm(dim)
        self.expander_teacher = nn.Linear(dim*rep_seeds_num, expanded_dim)

        self.student_list = [self.blocks, self.norm, self.expander]
        self.teacher_list = [self.blocks_teacher, self.norm_teacher, self.expander_teacher]

        self.flag_teacher_init = False

        self.distortor = InfoDistortor(
            dt=0.1,
            hist_len=21,
            low=[-1.0, -0.75, -0.35, -1, -0.5, -0.2, -0.1],
            high=[1.0, 0.75, 0.35, 1, 0.5, 0.2, 0.1],
            augment_prob=0.5,
            normalize=True,
        )
        

        self.trajectory_decoder = TrajectoryDecoder(
            embed_dim=dim,
            num_modes=num_modes,
            future_steps=future_steps,
            out_channels=4,
        )

        self.randint = torch.randint(0, self.N_mask, [256, 1024])

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

    def get_pretrain_modules(self):
        # targeted_list = [self.pos_emb, self.tempo_net, self.agent_seed, self.agent_projector, 
        #         self.map_projector, self.lane_pred, self.agent_frame_pred, self.blocks, self.norm]
        # module_list = [[name, module] for name, module in self.named_modules() if module in targeted_list]
        return [self.pos_emb, self.tempo_net, self.agent_seed, self.agent_projector, 
                self.map_projector, self.lane_pred, self.agent_frame_pred, self.blocks, self.norm, self.agent_predictor]

    def get_finetune_modules(self):
        return [self.expander, self.rep_seed, self.trajectory_decoder, self.plan_seed]


    def get_stage(self, current_epoch):
        # return Stage.FINE_TUNING
        if current_epoch < self.pretrain_epoch_stages[1]:
            return Stage.PRETRAIN_SEP
        # elif current_epoch < self.pretrain_epoch_stages[2]:
        #     if not self.flag_teacher_init:
        #         self.initialize_teacher()
        #         self.flag_teacher_init = True
        #     return Stage.PRETRAIN_REPRESENTATION
        elif current_epoch < self.pretrain_epoch_stages[2]:
            return Stage.FINETUNE
        else:
            return Stage.ANT_MASK_FINETUNE
        
        # for debugging
        

    def forward(self, data, current_epoch=None):
        if current_epoch is None: # when inference
            # return self.forward_pretrain_separate(data)
            return self.forward_finetune(data)
        else:
            stage = self.get_stage(current_epoch)
            if stage == Stage.PRETRAIN_SEP:
                return self.forward_pretrain_separate(data)
            elif stage == Stage.PRETRAIN_MIX:
                return self.forward_pretrain_mix(data)
            elif stage == Stage.PRETRAIN_REPRESENTATION:
                # self.EMA_update() # currently this is done in lightning_trainer.py
                return self.forward_pretrain_representation(data)
            elif stage == Stage.FINETUNE:
                return self.forward_finetune(data)
            elif stage == Stage.ANT_MASK_FINETUNE:
                return self.forward_antagonistic_mask_finetune(data)
            else:
                raise NotImplementedError(f"Stage {stage} is not implemented.")

    def extract_map_feature(self, data):
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
        # point_orientation = data["map"]["point_orientation"]

        # # B M 20
        valid_mask = data["map"]["valid_mask"]

        # point_position_feature = torch.zeros(point_position[:,:,0].shape, device=point_position.device) # B M 20 2
        # point_position_feature[valid_mask] = point_position[:,:,0][valid_mask]

        point_position_feature, valid_mask, new_poly_prop = PlanningModel.split_lane_segment(point_position[:,:,0], valid_mask, self.lane_split_threshold, polygon_property)
        point_position_feature = rearrange(point_position_feature, 'b m p c -> b m (p c)')

        feature = torch.cat([point_position_feature, new_poly_prop], dim=-1)
        polygon_key_padding = ~(valid_mask.any(-1))

        return feature, valid_mask, polygon_key_padding



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

 

    def extract_agent_feature(self, data, include_future=False):

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

        frame_feature = torch.cat([position, heading.unsqueeze(-1), velocity, shape], dim=-1)
        # frame_feature = rearrange(frame_feature, 'b a t c -> b a (t c)')
        category_rep = repeat(category, 'b a -> b a t d', t=steps, d = 1)
        feature = torch.cat([frame_feature, category_rep], dim=-1)
        agent_key_padding = ~(valid_mask.any(-1))

        return feature, valid_mask, agent_key_padding
        

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
            x_masked[ids_masked, 2:] = 0 # NOTE: keep polygon_type, polygon_on_route, polygon_tl_status, and the coords of starting point

            x_masked_list.append(x_masked)
            # new_key_padding_mask.append(torch.zeros(len_keep, device=x.device))
 
        x_masked_list = pad_sequence(x_masked_list, batch_first=True)
        # new_key_padding_mask = pad_sequence(
        #     new_key_padding_mask, batch_first=True, padding_value=True
        # )

        return x_masked_list, ids_keep_list


    def trajectory_random_masking(self, x, future_mask_ratio, frame_valid_mask, seed):
        '''
        x: (B, A, T, D). 
        future_mask_ratio: float
        key_padding_mask: (B, A, T)
        seed: (D, )

        each history consists of T frames, but not all frames are valid
        we first randomly masked out future_mask_ratio of the frames, but there is a possibility that all valid frames are masked
        therefore we manually give at least one valid frame to keep
        '''
        len_keep = math.ceil(self.history_steps * (1 - future_mask_ratio))

        noise = torch.rand(frame_valid_mask.shape[:3], device=x.device)
        sorted = torch.sort(noise)[0]

         # 1 indicated kept, 0 indicated masked
        kept_mask = noise < sorted[..., len_keep].unsqueeze(-1)
        noise_valid = noise * frame_valid_mask
        kept_mask[noise_valid.max(-1)==noise] = True # ensure that at least one valid frame is kept

        pred_mask = frame_valid_mask*~kept_mask

        # generate the masked tokens
        x_masked = x*kept_mask.unsqueeze(-1) + (~kept_mask.unsqueeze(-1))*repeat(seed, 'd -> b a t d', b=x.shape[0], a=x.shape[1], t=x.shape[2])

        return x_masked, pred_mask
    


    def forward_pretrain_separate(self, data):

        ## 1. MRM
        map_features, polygon_mask, polygon_key_padding = self.extract_map_feature(data)
        
        (
            lane_masked_tokens,
            lane_ids_keep_list,
        ) = self.lane_random_masking(
            map_features, self.lane_mask_ratio, polygon_key_padding
        )

        lane_embedding = self.map_projector(lane_masked_tokens)

        # transformer stack
        x = lane_embedding
        for blk in self.blocks:
            x = blk(x, key_padding_mask=polygon_key_padding)
        x = self.norm(x)
  
        # lane pred loss
        lane_pred_mask = polygon_mask.clone().detach() # attention
        for i, idx in enumerate(lane_ids_keep_list):
            lane_pred_mask[i, idx] = False

        lane_pred = rearrange(self.lane_pred(x), 'b n (p c) -> b n p c', p=self.no_lane_segment_points, c=2)
        # lane_reg_mask = ~polygon_mask
        # lane_reg_mask[~lane_pred_mask] = False
        lane_pred_loss = F.smooth_l1_loss(
            lane_pred[lane_pred_mask], rearrange(map_features[..., :self.no_lane_segment_points*2], 'b n (p c) -> b n p c', p=self.no_lane_segment_points, c=2)[lane_pred_mask], reduction='mean'
        )

        ## 2. MTM

        agent_features, frame_valid_mask, agent_key_padding = self.extract_agent_feature(data, include_future=True)

        agent_embedding = self.agent_projector(agent_features) # B A D
        (agent_masked_tokens, frame_pred_mask) = self.trajectory_random_masking(agent_embedding, self.trajectory_mask_ratio, frame_valid_mask, seed=self.agent_seed)

        agent_masked_tokens_ = rearrange(agent_masked_tokens, 'b a t d -> (b a) t d').clone()
        agent_masked_tokens_pos_embeded = self.pe(agent_masked_tokens_)
        # agent_tempo_key_padding = rearrange(~agent_mask, 'b a t -> (b a) t') 

        y = self.tempo_net(agent_masked_tokens_pos_embeded) # if key_padding_mask should be used here? this causes nan values in loss and needs investigation
        # frame pred loss
        frame_pred = rearrange(self.agent_frame_pred(y), '(b a) t c -> b a t c', b=agent_features.shape[0], a=agent_features.shape[1])
        agent_pred_loss = F.smooth_l1_loss(
            frame_pred[frame_pred_mask], agent_features[frame_pred_mask][..., :3]
        )

        # 3. TP
        bs, A = data["agent"]["heading"].shape[0:2]

        agent_embedding = self.agent_projector(agent_features[:,:,:self.history_steps]) # B A T D -> B A D
        agent_embedding_tempo = self.tempo_net(rearrange(agent_embedding, 'b a t d -> (b a) t d')) # if key_padding_mask should be used here? this causes nan values in loss and needs investigation
        agent_embedding_tempo = rearrange(agent_embedding_tempo, '(b a) t c -> b a t c', b=bs, a=A)
        agent_embedding_tempo = reduce(agent_embedding_tempo, 'b a t c -> b a c', 'max')
        lane_embedding = self.map_projector(map_features)
        concat = torch.cat([agent_embedding_tempo, lane_embedding], dim=1)

        agent_key_padding = ~(frame_valid_mask.any(-1))
        polygon_key_padding = ~(polygon_mask.any(-1))
        mask_concat = torch.cat([agent_key_padding, polygon_key_padding], dim=1)
        x = concat
        for blk in self.blocks:
            x = blk(x, key_padding_mask=mask_concat)
        x = self.norm(x)
        tail_prediction = self.agent_predictor(x[:, :A]).view(bs, -1, self.future_steps, 2)
        tail_mask = torch.ones_like(frame_valid_mask, dtype=torch.bool)
        tail_mask[:, :, :self.history_steps] = False
        tail_pred_mask = tail_mask & frame_valid_mask
        tail_pred_loss = F.smooth_l1_loss(
            tail_prediction[tail_pred_mask[:, :, self.history_steps:]], agent_features[tail_pred_mask][..., :2]
        )

        out = {
            "MRM_loss": lane_pred_loss,
            "MTM_loss": agent_pred_loss,
            "TP_loss": tail_pred_loss,
            "loss": lane_pred_loss + agent_pred_loss + tail_pred_loss,
        }

        return out

    def plot_scene_attention(self, data, attn_weights, output_trajectory, key_padding_mask, k=0):
        i = 0
        map_features, polygon_mask, polygon_key_padding = self.extract_map_feature(data)
        agent_features, agent_mask, agent_key_padding = self.extract_agent_feature(data, include_future=False)
        assert agent_features.shape[1]+map_features.shape[1] == attn_weights.shape[1]-(1+self.rep_seeds_num)
        map_points = map_features[i][..., :40]
        map_points_reshape = map_points.reshape(map_points.shape[0], -1, 2)
        plot_scene_attention(agent_features[i], agent_mask[i], map_points_reshape, attn_weights[i, (1+self.rep_seeds_num):],
                             key_padding_mask[i, (1+self.rep_seeds_num):], 
                              output_trajectory[i], filename=self.inference_counter, prefix=k)
        

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

    def generate_antagonistic_masks(self, key_padding_mask):
        '''
        This function generates N_mask antagonistic masks. 
        All the masks belong to one set should sum up to the key_padding_mask corresponding to that scene. 
        All the masks should contain roughly the similar number of preserved values.

        key_padding_mask: (B, N, M)
        '''

        key_padding_mask_sliced = key_padding_mask[:, (1+self.rep_seeds_num):].clone()
        randint = torch.randint(0, self.N_mask, key_padding_mask_sliced.shape, device=key_padding_mask_sliced.device)
        masks = torch.zeros((self.N_mask, key_padding_mask_sliced.shape[0], key_padding_mask_sliced.shape[1]),
                             device=key_padding_mask_sliced.device, dtype=torch.bool)
        for i in range(self.N_mask):
            # element being 1 in mask should be kept
            masks[i] = key_padding_mask_sliced | (~(randint==i))

        assert ((~masks).sum(0) == ~key_padding_mask_sliced).all() # all sum up to 1

        # the following part will cause CUDA assert error and is not necessary in principle
        # if not (~masks).sum(-1).all():
        #     print('There exists a mask that does not contain any preserved value')
        #     problem_pos = torch.nonzero((~masks).sum(-1)==0)
        #     for i in problem_pos.shape[0]:
        #         masks[problem_pos[i]] = key_padding_mask_sliced[problem_pos[i,-1]] # keep the original key_padding_mask
        #     print('Successfully dealt')
        
        return masks.permute(1, 0, 2)

    def forward_finetune(self, data):
        bs, A = data["agent"]["heading"].shape[0:2]
        x_orig, key_padding_mask = self.embed(data, torch.cat((self.plan_seed, self.rep_seed)))

        x = x_orig 
        for blk in self.blocks:
            x = blk(x, key_padding_mask=key_padding_mask)
        x = self.norm(x)

        trajectory, probability = self.trajectory_decoder(x[:, 0])
        prediction = self.agent_predictor(x[:, 1+(1+self.rep_seeds_num):A+(1+self.rep_seeds_num)]).view(bs, -1, self.future_steps, 2)

        out = {
            "trajectory": trajectory,
            "probability": probability,
            "prediction": prediction.unsqueeze(1),
        }

        if not self.training:
            best_mode = probability.argmax(dim=-1)
            output_trajectory = trajectory[torch.arange(bs), best_mode]
            angle = torch.atan2(output_trajectory[..., 3], output_trajectory[..., 2])
            out["output_trajectory"] = torch.cat(
                [output_trajectory[..., :2], angle.unsqueeze(-1)], dim=-1
            )

        # attention visualization
        if False:
            attn_weights = self.blocks[-1].attn_mat[:, 0].detach()
            # visualize the scene using the attention weights
            self.plot_scene_attention(data, attn_weights, output_trajectory, key_padding_mask, 0)
            self.inference_counter += 1

        return out


    def forward_antagonistic_mask_finetune(self, data):
        bs, A = data["agent"]["heading"].shape[0:2]
        x_orig, key_padding_mask = self.embed(data, torch.cat((self.plan_seed, self.rep_seed)))

        masks = self.generate_antagonistic_masks(key_padding_mask) # B, N_mask, M

        masks = rearrange(masks, 'b n m -> (b n) m')
        masks = torch.cat([torch.zeros([masks.shape[0], 1+self.rep_seeds_num], dtype=torch.bool, device=masks.device), masks], dim=1)
        x = repeat(x_orig, 'b m d -> (b n) m d', n=self.N_mask)

        # x = x_orig # comment this if not debugging!
        # masks = masks[0]
        # masks = torch.cat([torch.zeros([masks.shape[0], 1+self.rep_seeds_num], dtype=torch.bool, device=masks.device), masks], dim=1)
        for blk in self.blocks:
            x = blk(x, key_padding_mask=masks)
        x = self.norm(x)
        # attn_weights = self.blocks[-1].attn_mat[:, 0].detach()

        trajectory, probability = self.trajectory_decoder(x[:, 0])
        prediction = self.agent_predictor(x[:, 1+(1+self.rep_seeds_num):A+(1+self.rep_seeds_num)]).view(bs, -1,  A-1, self.future_steps, 2)

        trajectory = rearrange(trajectory, '(b n) m t c -> b (n m) t c', n=self.N_mask)
        probability = rearrange(probability, '(b n) m -> b (n m)', n=self.N_mask)

        out = {
                "trajectory": trajectory,
                "probability": probability,
                "prediction": prediction,
            }
        
        if not self.training:
                
            x = x_orig.detach().clone()
            for blk in self.blocks:
                x = blk(x, key_padding_mask=key_padding_mask)
            x = self.norm(x)

            trajectory, probability = self.trajectory_decoder(x[:, 0])

            best_mode = probability.argmax(dim=-1)
            output_trajectory = trajectory[torch.arange(bs), best_mode]
            angle = torch.atan2(output_trajectory[..., 3], output_trajectory[..., 2])
            out["output_trajectory"] = torch.cat(
                [output_trajectory[..., :2], angle.unsqueeze(-1)], dim=-1
            )

        # self.inference_counter += 1

        return out
    
    
    def embed(self, data, seeds, include_future=False):
        """
            data: dict
            seeds: tensor (D, )
        """
        
        map_features, polygon_mask, polygon_key_padding = self.extract_map_feature(data)
        agent_features, agent_mask, agent_key_padding = self.extract_agent_feature(data, include_future)

        bs, A = agent_features.shape[0:2]
        if seeds.dim() == 1:
            seeds = repeat(seeds, 'd -> bs 1 d', bs=bs)
        else:
            seeds = repeat(seeds, 'n d -> bs n d', bs=bs)    

        # lane embedding (purely projection)
        lane_embedding = self.map_projector(map_features)

        # agent embedding
        agent_embedding = rearrange(self.agent_projector(agent_features), 'b a t d -> (b a) t d').clone()
        agent_embedding[~rearrange(agent_mask,'b a t -> (b a) t')] = self.agent_seed
        agent_embedding_pos_embed = self.pe(agent_embedding)
        # agent_tempo_key_padding = rearrange(~agent_mask, 'b a t -> (b a) t')
        x_agent = self.tempo_net(agent_embedding_pos_embed) # NOTE
        x_agent_ = rearrange(x_agent, '(b a) t c -> b a t c', b=agent_features.shape[0], a=agent_features.shape[1])
        x_agent_ = reduce(x_agent_, 'b a t c -> b a c', 'max')

        x = torch.cat([seeds, x_agent_, lane_embedding], dim=1)

        # key padding masks
        key_padding_mask = torch.cat([torch.zeros(seeds.shape[0:2], device=agent_key_padding.device, dtype=torch.bool), agent_key_padding, polygon_key_padding], dim=-1)

        return x, key_padding_mask

    def mask_and_embed(self, data, seeds):
        """
            data: dict
            seeds: tensor (D, )
        """

        agent_features, agent_mask, agent_key_padding = self.extract_agent_feature(data)
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

        lane_embedding = self.map_projector(lane_masked_tokens)

        agent_embedding = self.agent_projector(agent_features) # B A D
        (agent_masked_tokens, frame_pred_mask) = self.trajectory_random_masking(agent_embedding, self.trajectory_mask_ratio, agent_mask, seed=self.agent_seed)

        b, a = agent_masked_tokens.shape[0:2]
        agent_masked_tokens_ = rearrange(agent_masked_tokens, 'b a t d -> (b a) t d').clone()
        agent_masked_tokens_pos_embeded = self.pe(agent_masked_tokens_)
        # agent_tempo_key_padding = rearrange(~agent_mask, 'b a t -> (b a) t')

        x_agent = self.tempo_net(agent_masked_tokens_pos_embeded)
        x_agent_ = rearrange(x_agent, '(b a) t c -> b a t c', b=b, a=a)
        x_agent_ = reduce(x_agent_, 'b a t c -> b a c', 'max')

        x = torch.cat([seeds, x_agent_, lane_embedding], dim=1)

        # key padding masks
        agent_key_padding = ~(agent_mask.any(-1))
        polygon_key_padding = ~(polygon_mask.any(-1))
        key_padding_mask = torch.cat([torch.zeros(seeds.shape[0:2], device=agent_key_padding.device, dtype=torch.bool), agent_key_padding, polygon_key_padding], dim=-1)

        return x, key_padding_mask

    def forward_pretrain_representation(self, data):
        data_distorted = self.distortor.augment(data)
        x_stu, x_stu_key_padding_mask = self.mask_and_embed(data_distorted, self.rep_seed)

        # forward through student model 
        for blk in self.blocks:
            x_stu = blk(x_stu, key_padding_mask=x_stu_key_padding_mask)
        x_stu = self.norm(x_stu)

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

