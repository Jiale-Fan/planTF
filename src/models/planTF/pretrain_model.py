from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from .modules.transformer_blocks import Block
from .modules.trajectory_decoder import TrajectoryDecoder


class PretrainModel(nn.Module):
    def __init__(
        self,
        embed_dim=128,
        encoder_depth=4,
        decoder_depth=4,
        num_heads=8,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop_path=0.2,
        actor_mask_ratio: float = 0.5,
        lane_mask_ratio: float = 0.5,
        history_steps: int = 50,
        future_steps: int = 60,
        loss_weight: List[float] = [1.0, 1.0, 0.35],
        pred_modes: int = 6,
    ) -> None:
        super().__init__()

        self.embed_dim = embed_dim
        self.actor_mask_ratio = actor_mask_ratio
        self.lane_mask_ratio = lane_mask_ratio
        self.loss_weight = loss_weight
        self.history_steps = history_steps
        self.future_steps = future_steps
        self.decoder_depth = decoder_depth
        self.num_modes = pred_modes

        self.pos_embed = nn.Sequential(
            nn.Linear(4, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        dpr = [x.item() for x in torch.linspace(0, drop_path, encoder_depth)]
        self.blocks = nn.ModuleList(
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop_path=dpr[i],
            )
            for i in range(encoder_depth)
        )
        self.norm = nn.LayerNorm(embed_dim)

        # decoder
        self.decoder_embed = nn.Linear(embed_dim, embed_dim, bias=True)
        self.decoder_pos_embed = nn.Sequential(
            nn.Linear(4, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

        dpr = [x.item() for x in torch.linspace(0, drop_path, decoder_depth)]
        self.decoder_blocks = nn.ModuleList(
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop_path=dpr[i],
            )
            for i in range(decoder_depth)
        )
        self.decoder_norm = nn.LayerNorm(embed_dim)

        self.lane_mask_token = nn.Parameter(torch.Tensor(1, 1, embed_dim))
        self.future_mask_token = nn.Parameter(torch.Tensor(1, 1, embed_dim))
        self.history_mask_token = nn.Parameter(torch.Tensor(1, 1, embed_dim))

        # self.future_pred = nn.Linear(embed_dim, future_steps * 2)
        # self.history_pred = nn.Linear(embed_dim, history_steps * 2)
        # self.lane_pred = nn.Linear(embed_dim, 20 * 2)

        self.future_pred = TrajectoryDecoder(embed_dim, pred_modes, future_steps, 2)
        self.history_pred = TrajectoryDecoder(embed_dim, pred_modes, history_steps, 2)
        self.lane_pred = nn.Linear(embed_dim, 20 * 2)

        self.initialize_weights()

    def initialize_weights(self):
        nn.init.normal_(self.future_mask_token, std=0.02)
        nn.init.normal_(self.lane_mask_token, std=0.02)
        nn.init.normal_(self.history_mask_token, std=0.02)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @staticmethod
    def agent_random_masking(
        hist_tokens, fut_tokens, mask_ratio, future_padding_mask, num_actors
    ):
        pred_masks = ~future_padding_mask.all(-1)  # [B, A]
        fut_num_tokens = pred_masks.sum(-1)  # [B]

        len_keeps = (fut_num_tokens * (1 - mask_ratio)).int()
        hist_masked_tokens, fut_masked_tokens = [], []
        hist_keep_ids_list, fut_keep_ids_list = [], []
        hist_key_padding_mask, fut_key_padding_mask = [], []

        device = hist_tokens.device
        agent_ids = torch.arange(hist_tokens.shape[1], device=device)
        for i, (fut_num_token, len_keep, future_pred_mask) in enumerate(
            zip(fut_num_tokens, len_keeps, pred_masks)
        ):
            pred_agent_ids = agent_ids[future_pred_mask]
            noise = torch.rand(fut_num_token, device=device)
            ids_shuffle = torch.argsort(noise)
            fut_ids_keep = ids_shuffle[:len_keep]
            fut_ids_keep = pred_agent_ids[fut_ids_keep]
            fut_keep_ids_list.append(fut_ids_keep)

            hist_keep_mask = torch.zeros_like(agent_ids).bool()
            hist_keep_mask[: num_actors[i]] = True
            hist_keep_mask[fut_ids_keep] = False
            hist_ids_keep = agent_ids[hist_keep_mask]
            hist_keep_ids_list.append(hist_ids_keep)

            fut_masked_tokens.append(fut_tokens[i, fut_ids_keep])
            hist_masked_tokens.append(hist_tokens[i, hist_ids_keep])

            fut_key_padding_mask.append(torch.zeros(len_keep, device=device))
            hist_key_padding_mask.append(torch.zeros(len(hist_ids_keep), device=device))

        fut_masked_tokens = pad_sequence(fut_masked_tokens, batch_first=True)
        hist_masked_tokens = pad_sequence(hist_masked_tokens, batch_first=True)
        fut_key_padding_mask = pad_sequence(
            fut_key_padding_mask, batch_first=True, padding_value=True
        )
        hist_key_padding_mask = pad_sequence(
            hist_key_padding_mask, batch_first=True, padding_value=True
        )

        return (
            hist_masked_tokens,
            hist_keep_ids_list,
            hist_key_padding_mask,
            fut_masked_tokens,
            fut_keep_ids_list,
            fut_key_padding_mask,
        )

    @staticmethod
    def lane_random_masking(x, future_mask_ratio, key_padding_mask):
        num_tokens = (~key_padding_mask).sum(1)  # (B, )
        len_keeps = torch.ceil(num_tokens * (1 - future_mask_ratio)).int()

        x_masked, new_key_padding_mask, ids_keep_list = [], [], []
        for i, (num_token, len_keep) in enumerate(zip(num_tokens, len_keeps)):
            noise = torch.rand(num_token, device=x.device)
            ids_shuffle = torch.argsort(noise)

            ids_keep = ids_shuffle[:len_keep]
            ids_keep_list.append(ids_keep)
            x_masked.append(x[i, ids_keep])
            new_key_padding_mask.append(torch.zeros(len_keep, device=x.device))

        x_masked = pad_sequence(x_masked, batch_first=True)
        new_key_padding_mask = pad_sequence(
            new_key_padding_mask, batch_first=True, padding_value=True
        )

        return x_masked, new_key_padding_mask, ids_keep_list

    def forward(self, hist_feat, lane_feat, future_feat, hist_mask, fut_mask, lane_padding_mask, pos_feat,
                lane_normalized, hist_target, fut_target, types_embedding):
        """
        Args:
            hist_feat: (B, N, C)
            lane_feat: (B, M, C)
            future_feat: (B, N, C)
            hist_padding_mask: (B, N, hist_steps), 0 indicates padding
            fut_padding_mask: (B, N, future_steps)
            lane_padding_mask: (B, M)
            pos_feat: (B, 2*N+M, 4)
            lane_normalized: (B, M, 20, 2)
            hist_target: (B, N, hist_steps, 2)
            fut_target: (B, N, fut_steps, 2)
            types_embedding: (B, 2*N+M, 4)
        """
        B, N, _ = hist_feat.shape
        _, M, _ = lane_feat.shape

        agent_padding_mask = ~(hist_mask.any(-1))
        fut_padding_mask = ~(fut_mask.any(-1))

        (
            hist_masked_tokens,
            hist_keep_ids_list,
            hist_key_padding_mask,
            fut_masked_tokens,
            fut_keep_ids_list,
            fut_key_padding_mask,
        ) = self.agent_random_masking(
            hist_feat,
            future_feat,
            self.actor_mask_ratio,
            ~fut_mask,
            num_actors=((~agent_padding_mask)|(~fut_padding_mask)).sum(-1),
        )

        lane_mask_ratio = self.lane_mask_ratio
        (
            lane_masked_tokens,
            lane_key_padding_mask,
            lane_ids_keep_list,
        ) = self.lane_random_masking(
            lane_feat, lane_mask_ratio, lane_padding_mask
        )

        x_target = torch.cat(
            [hist_masked_tokens, fut_masked_tokens, lane_masked_tokens], dim=1
        )
        key_padding_mask = torch.cat(
            [hist_key_padding_mask, fut_key_padding_mask, lane_key_padding_mask],
            dim=1,
        )

        for blk in self.blocks:
            x_target = blk(x_target, key_padding_mask=key_padding_mask)
        x_target = self.norm(x_target)

        # decoding
        x_decoder = self.decoder_embed(x_target)
        Nh, Nf, Nl = (
            hist_masked_tokens.shape[1],
            fut_masked_tokens.shape[1],
            lane_masked_tokens.shape[1],
        )
        assert x_decoder.shape[1] == Nh + Nf + Nl
        hist_tokens = x_decoder[:, :Nh]
        fut_tokens = x_decoder[:, Nh : Nh + Nf]
        lane_tokens = x_decoder[:, -Nl:]

        decoder_hist_token = self.history_mask_token.repeat(B, N, 1)
        hist_pred_mask = ~agent_padding_mask
        for i, idx in enumerate(hist_keep_ids_list):
            decoder_hist_token[i, idx] = hist_tokens[i, : len(idx)]
            hist_pred_mask[i, idx] = False

        decoder_fut_token = self.future_mask_token.repeat(B, N, 1)
        future_pred_mask = ~agent_padding_mask
        for i, idx in enumerate(fut_keep_ids_list):
            decoder_fut_token[i, idx] = fut_tokens[i, : len(idx)]
            future_pred_mask[i, idx] = False

        decoder_lane_token = self.lane_mask_token.repeat(B, M, 1)
        lane_pred_mask = ~lane_padding_mask
        for i, idx in enumerate(lane_ids_keep_list):
            decoder_lane_token[i, idx] = lane_tokens[i, : len(idx)]
            lane_pred_mask[i, idx] = False

        x_decoder = torch.cat(
            [decoder_hist_token, decoder_fut_token, decoder_lane_token], dim=1
        )
        x_decoder = x_decoder + self.decoder_pos_embed(pos_feat) + types_embedding
        decoder_key_padding_mask = torch.cat(
            [
                agent_padding_mask,
                fut_padding_mask,
                lane_padding_mask,
            ],
            dim=1,
        )

        for blk in self.decoder_blocks:
            x_decoder = blk(x_decoder, key_padding_mask=decoder_key_padding_mask)

        x_decoder = self.decoder_norm(x_decoder)
        hist_token = x_decoder[:, :N]
        future_token = x_decoder[:, N : 2 * N]
        lane_token = x_decoder[:, -M:]

        # lane pred loss
        lane_pred = self.lane_pred(lane_token).view(B, M, 20, 2)
        lane_reg_mask = ~lane_padding_mask
        lane_reg_mask[~lane_pred_mask] = False
        lane_pred_loss = F.mse_loss(
            lane_pred[lane_reg_mask], lane_normalized[lane_reg_mask]
        )

        # hist pred loss
        x_hat = self.history_pred(hist_token)[0] # (B, N, self.num_modes, self.history_steps, 2)
        diff_x = F.l1_loss(x_hat, hist_target.unsqueeze(2).repeat(1,1,self.num_modes,1,1), reduction='none').mean(-1)  # (B, N, self.num_modes, self.history_steps)
        diff_x_reshape = diff_x.view(-1, self.num_modes, self.history_steps).permute(1, 0, 2) # (self.num_modes, B*N, self.history_steps)
        x_reg_mask = hist_mask.clone().detach()
        x_reg_mask[~hist_pred_mask] = False
        x_reg_mask = x_reg_mask.view(-1, self.history_steps).unsqueeze(0).repeat(self.num_modes, 1, 1) # (self.num_modes, B*N, self.history_steps)
        diff_x_valid = (diff_x_reshape*x_reg_mask).sum(-1).permute(1, 0).min(-1)[0]
        hist_loss = diff_x_valid.sum()/x_reg_mask.sum()

        # future pred loss
        y_hat = self.future_pred(future_token)[0]
        diff_y = F.l1_loss(y_hat, fut_target.unsqueeze(2).repeat(1,1,self.num_modes,1,1), reduction='none').mean(-1)
        diff_y_reshape = diff_y.view(-1, self.num_modes, self.future_steps).permute(1, 0, 2)
        reg_mask = fut_mask.clone().detach()
        reg_mask[~future_pred_mask] = False
        reg_mask = reg_mask.view(-1, self.future_steps).unsqueeze(0).repeat(self.num_modes, 1, 1)
        diff_y_valid = (diff_y_reshape*reg_mask).sum(-1).permute(1, 0).min(-1)[0]
        future_loss = diff_y_valid.sum()/reg_mask.sum()

        loss = (
            self.loss_weight[0] * future_loss
            + self.loss_weight[1] * hist_loss
            + self.loss_weight[2] * lane_pred_loss
        )

        out = {
            "loss": loss,
            "hist_loss": hist_loss.item(),
            "future_loss": future_loss.item(),
            "lane_pred_loss": lane_pred_loss.item(),
        }

        if not self.training:
            out["x_hat"] = x_hat.view(B, N, self.history_steps, 2)
            out["y_hat"] = y_hat.view(1, B, N, self.future_steps, 2)
            out["lane_hat"] = lane_pred.view(B, M, 20, 2)
            out["lane_keep_ids"] = lane_ids_keep_list
            out["hist_keep_ids"] = hist_keep_ids_list
            out["fut_keep_ids"] = fut_keep_ids_list

        return out