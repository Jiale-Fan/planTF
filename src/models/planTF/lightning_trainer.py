import logging
import os
from typing import Dict, Tuple, Union

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from nuplan.planning.training.modeling.torch_module_wrapper import TorchModuleWrapper
from nuplan.planning.training.modeling.types import (
    FeaturesType,
    ScenarioListType,
    TargetsType,
)
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torchmetrics import MetricCollection

from src.metrics import MR, minADE, minFDE
from src.optim.warmup_cos_lr import WarmupCosLR
from src.models.planTF.training_objectives import nll_loss_multimodes_joint
from src.models.planTF.pairing_matrix import proj_name_to_mat

from torch.nn.utils.rnn import pad_sequence

logger = logging.getLogger(__name__)


class LightningTrainer(pl.LightningModule):
    def __init__(
        self,
        model: TorchModuleWrapper,
        lr,
        weight_decay,
        epochs,
        warmup_epochs,
        modes_contrastive_weight = 10.0, # 100
        scenario_type_contrastive_weight = 50,
        contrastive_temperature = 0.3,
        modes_contrastive_negative_threshold = 2.0,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["model"])

        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.warmup_epochs = warmup_epochs
        self.temperature = contrastive_temperature # TODO: adjust temperature?

        self.modes_contrastive_weight = modes_contrastive_weight # TODO: adjust contrastive_weight?
        self.scenario_type_contrastive_weight = scenario_type_contrastive_weight
        self.modes_contrastive_negative_threshold = modes_contrastive_negative_threshold


    def on_fit_start(self) -> None:
        # self.model.train()
        metrics_collection = MetricCollection(
            {
                "minADE1": minADE(k=1).to(self.device),
                "minADE6": minADE(k=6).to(self.device),
                "minFDE1": minFDE(k=1).to(self.device),
                "minFDE6": minFDE(k=6).to(self.device),
                "MR": MR().to(self.device),
            }
        )
        self.metrics = {
            "train": metrics_collection.clone(prefix="train/"),
            "val": metrics_collection.clone(prefix="val/"),
        }

    def _step(
        self, batch: Tuple[FeaturesType, TargetsType, ScenarioListType], prefix: str
    ) -> torch.Tensor:
        features, _, _ = batch
        res = self.forward(features["feature"].data)

        losses = self._compute_objectives(res, features["feature"].data)
        metrics = self._compute_metrics(res, features["feature"].data, prefix)
        self._log_step(losses["loss"], losses, metrics, prefix)

        return losses["loss"]

    def _compute_objectives(self, res, data) -> Dict[str, torch.Tensor]:
        probability, prediction = (
            res["probability"], # [batch, num_modes, num_agents, time_steps, 5]
            res["prediction"], # [batch, num_modes]
        )

        targets = data["agent"]["target"]
        valid_mask = data["agent"]["valid_mask"][..., -targets.shape[-2]:]

        nll_loss, kl_loss, post_entropy, adefde_loss = \
            nll_loss_multimodes_joint(prediction.permute(1,3,0,2,4), targets.permute(0,2,1,3), probability, valid_mask.permute(0,2,1),
                                        entropy_weight=40.0,
                                        kl_weight=20.0,
                                        use_FDEADE_aux_loss=True,
                                        predict_yaw=True)
        
        # contrastive loss
        # 1. contrastive loss between scene understandings
        # look up tables to get positive and negative pairs TODO debug; examine if MASKS are needed!

        scene_type_loss_dict ={}
        scene_type_loss_sum = 0

        for proj_name in ["beh_proj", "env_proj", "obj_proj"]:
            query_embs, pos_embs, neg_embs, neg_masks = \
                self.get_positive_negative_embs_scenario_types(targets[:, 0], data["scenario_type"], res[proj_name], proj_name)
            contrastive_loss_type = self._contrastive_loss(query_embs, pos_embs, neg_embs, neg_masks) \
                                    if pos_embs is not None else 0
            scene_type_loss_dict[proj_name] = contrastive_loss_type
            scene_type_loss_sum += contrastive_loss_type

        # 2. contrastive loss between multi-modal plans
        contrastive_loss_modes = 0
        if self.training:
            neg_masks = self.get_negative_embs_masks(res["trajectory"], targets[:, 0]) 
            contrastive_loss_modes = self._contrastive_loss(res["scene_best_emb_proj"], 
                                                            res["scene_target_emb_proj"],
                                                            res["scene_plan_emb_proj"], neg_masks)

        if self.current_epoch < 10:
            loss = nll_loss + adefde_loss + kl_loss + \
                 1e-10 * contrastive_loss_modes + \
                 1e-10 * scene_type_loss_sum
        else:
            loss = nll_loss + adefde_loss + kl_loss + \
                 self.modes_contrastive_weight * contrastive_loss_modes + \
                 self.scenario_type_contrastive_weight * scene_type_loss_sum

        return {
            "loss": loss,

            "kl_loss": kl_loss,
            "post_entropy": post_entropy,
            "ade_fde_loss": adefde_loss,
            "nll_loss": nll_loss,

            "contrastive_loss": contrastive_loss_modes,
            "beh_contrastive_loss": scene_type_loss_dict["beh_proj"],
            "env_contrastive_loss": scene_type_loss_dict["env_proj"],
            "obj_contrastive_loss": scene_type_loss_dict["obj_proj"],
        }

    def get_negative_embs_masks(self, multimodal_trajs, ego_target):
        '''
        Input:
            multimodal_trajs: [batch, num_modes, time_steps, 6]
            ego_target: [batch, time_steps, 2]
        Output:
            neg_masks: [batch, num_modes]
        '''
        # calculate the ade and fde of each mode
        bs = multimodal_trajs.size(0)
        multimodal_trajs = multimodal_trajs.view((-1, multimodal_trajs.size(-2), multimodal_trajs.size(-1))).unsqueeze(0).repeat(bs, 1, 1, 1)  # [batch, batch*num_modes, time_steps, 6]
        fde = torch.norm(multimodal_trajs[:, :, -1, :2] - ego_target[:, None, -1, :2], dim=-1) # [batch, batch*num_modes]

        neg_masks = (fde > self.modes_contrastive_negative_threshold).to(torch.bool) # [batch, batch*num_modes]
        return neg_masks


    def get_positive_negative_embs_scenario_types(self, ego_targets, scenario_types_ids, projections, proj_name):
        '''
        Input:
            ego_targets: [batch_size, time_steps, 3]
            scenario_types_ids: [batch_size]
            projections: [batch_size, proj_dim(8)]
        Output:
            pos_embs: [batch_size, dim]
            neg_embs: [batch_size, num_neg, dim]
            neg_masks: [batch_size, num_neg]
        '''

        bs = scenario_types_ids.size(0)
        scenario_types_ids = scenario_types_ids.to(torch.long)
        # get positive and negative pairs of behavior embeddings
        pairing_mat = proj_name_to_mat[proj_name][scenario_types_ids[:, None], scenario_types_ids].squeeze()*(~torch.eye(bs).to(torch.bool)) # exclude self
        
        # exclude rows where there is no positive pair or no negative pair
        valid_idx = torch.nonzero(torch.any(pairing_mat>0, dim=-1) & torch.any(pairing_mat<0, dim=-1), as_tuple=True)[0]
        pairing_mat_valid = pairing_mat[valid_idx] # [batch_valid, batch]
        # pos_embs = pad_sequence([projections[torch.where(pairing_mat_valid[i]>0)] for i in range(bs)], batch_first=True) # [batch, pairs, dim]
        # neg_embs = pad_sequence([projections[torch.where(pairing_mat_valid[i]<0)] for i in range(bs)], batch_first=True)

        if valid_idx.size(0) == 0:
            return None, None, None, None
        else: 
            neg_embs = projections.unsqueeze(0).repeat(valid_idx.size(0), 1, 1) # [batch_valid, batch, dim]
            neg_masks = pairing_mat_valid < 0 # [batch_valid, batch]
            neg_masks = neg_masks.to(projections.device)

            errors_mat = (ego_targets[valid_idx, None, :, :2] - ego_targets[None, valid_idx, :, :2]).norm(dim=-1).sum(-1) # [batch_valid, batch_valid]
            errors_mat = errors_mat + torch.eye(errors_mat.size(0)).to(errors_mat.device)*1e6 # exclude self
            least_error = torch.argmin(errors_mat, dim=-1) # [batch_valid]

            pos_embs = projections[valid_idx[least_error]] # [batch_valid, dim]
            query_embs = projections[valid_idx] # [batch_valid, dim]
        
        return query_embs, pos_embs, neg_embs, neg_masks

    
    def _contrastive_loss(self, emb_query, emb_pos, emb_neg, neg_masks: torch.Tensor):
        '''
        Input:
            emb_query: [batch, dim]
            emb_pos: [batch, dim]
            emb_neg: [batch, num_neg, dim]
            neg_masks: [batch, num_neg]
        Output:
            loss: scalar
        '''

        emb_query = emb_query
        query = nn.functional.normalize(emb_query, dim=-1)

        # normalized embedding
        key_pos = nn.functional.normalize(emb_pos, dim=-1)
        key_neg = nn.functional.normalize(emb_neg, dim=-1)
        # pairing
        sim_pos = (query * key_pos).sum(dim=-1)
        sim_neg = (query[:, None, :] * key_neg).sum(dim=-1)

        # neg_masks = neg_masks.to(torch.bool)

        # bs = sim_neg.size(0)
        # loss = torch.zeros(bs)

        # for i in range(bs):
        #     sim_neg_i = sim_neg[i, torch.nonzero(neg_masks[i])].squeeze()
        #     if sim_neg_i.numel() > 0:
        #         loss[i] = -torch.log(torch.exp(sim_pos[i] / self.temperature) / torch.sum(torch.exp(sim_neg_i / self.temperature), dim=-1))

        # loss (social-nce)

        # logits = (torch.cat([sim_pos.unsqueeze(1), sim_neg], dim=1) / self.temperature)
        # labels = torch.zeros(logits.size(0), dtype=torch.long, device=self.device)
        # loss = self.contrastive_criterion(logits, labels)

        # loss
        # neg_masks.requires_grad_(False)
        denominator = torch.sum(torch.exp(sim_neg / self.temperature) * neg_masks, dim=-1)
        # pick the non-zero subset of denominator
        nonzero_deno = denominator[torch.nonzero(denominator).squeeze()]
        numerator = torch.exp(sim_pos / self.temperature)[torch.nonzero(denominator).squeeze()]
        loss = -torch.log(numerator/nonzero_deno).mean()

        return loss
    
    # def _contrastive_loss_multi(self, emb_query, emb_pos, emb_neg):
    #     '''
    #     Input:
    #         emb_query: [batch, dim]
    #         emb_pos: [batch, max_pairs, dim]
    #         emb_neg: [batch, max_pairs, dim]
    #         mask_pos: [batch, max_pairs]
    #         mask_neg: [batch, max_pairs]
    #     Output:
    #         loss: scalar
    #     '''

    #     emb_query = emb_query
    #     query = nn.functional.normalize(emb_query, dim=-1)

    #     # normalized embedding
    #     key_pos = nn.functional.normalize(emb_pos, dim=-1)
    #     key_neg = nn.functional.normalize(emb_neg, dim=-1)
    #     # pairing
    #     sim_pos = (query[:, None, :] * key_pos).sum(dim=-1)
    #     sim_neg = (query[:, None, :] * key_neg).sum(dim=-1) 
    #     # I think it is not necessary to exclude padding embeddings, since the padding embeddings are all zeros
    #     # and the dot product between query and zero embeddings would produce zero gradients for the query embedding

    #     # loss
    #     loss = -torch.log(torch.sum(torch.exp(sim_pos / self.temperature), dim=-1) / torch.sum(torch.exp(sim_neg / self.temperature), dim=-1)).mean()

    #     return loss


    def _compute_metrics(self, output, data, prefix) -> Dict[str, torch.Tensor]:
        metrics = self.metrics[prefix](output, data["agent"]["target"][:, 0])
        return metrics

    def _log_step(
        self,
        loss: torch.Tensor,
        objectives: Dict[str, torch.Tensor],
        metrics: Dict[str, torch.Tensor],
        prefix: str,
        loss_name: str = "loss",
    ) -> None:
        self.log(
            f"loss/{prefix}_{loss_name}",
            loss,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )

        for key, value in objectives.items():
            self.log(
                f"objectives/{prefix}_{key}",
                value,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )

        if metrics is not None:
            self.log_dict(
                metrics,
                prog_bar=(prefix == "val"),
                on_step=False,
                on_epoch=True,
                batch_size=1,
                sync_dist=True,
            )

    def training_step(
        self, batch: Tuple[FeaturesType, TargetsType, ScenarioListType], batch_idx: int
    ) -> torch.Tensor:
        """
        Step called for each batch example during training.

        :param batch: example batch
        :param batch_idx: batch's index (unused)
        :return: model's loss tensor
        """
        return self._step(batch, "train")

    def validation_step(
        self, batch: Tuple[FeaturesType, TargetsType, ScenarioListType], batch_idx: int
    ) -> torch.Tensor:
        """
        Step called for each batch example during validation.

        :param batch: example batch
        :param batch_idx: batch's index (unused)
        :return: model's loss tensor
        """
        return self._step(batch, "val")

    def test_step(
        self, batch: Tuple[FeaturesType, TargetsType, ScenarioListType], batch_idx: int
    ) -> torch.Tensor:
        """
        Step called for each batch example during testing.

        :param batch: example batch
        :param batch_idx: batch's index (unused)
        :return: model's loss tensor
        """
        return self._step(batch, "test")

    def forward(self, features: FeaturesType) -> TargetsType:
        """
        Propagates a batch of features through the model.

        :param features: features batch
        :return: model's predictions
        """
        return self.model(features)

    def configure_optimizers(
        self,
    ) -> Union[Optimizer, Dict[str, Union[Optimizer, _LRScheduler]]]:
        """
        Configures the optimizers and learning schedules for the training.

        :return: optimizer or dictionary of optimizers and schedules
        """
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (
            nn.Linear,
            nn.Conv1d,
            nn.Conv2d,
            nn.Conv3d,
            nn.MultiheadAttention,
            nn.LSTM,
            nn.GRU,
        )
        blacklist_weight_modules = (
            nn.BatchNorm1d,
            nn.BatchNorm2d,
            nn.BatchNorm3d,
            nn.SyncBatchNorm,
            nn.LayerNorm,
            nn.Embedding,
        )
        for module_name, module in self.named_modules():
            for param_name, param in module.named_parameters():
                full_param_name = (
                    "%s.%s" % (module_name, param_name) if module_name else param_name
                )
                if "bias" in param_name:
                    no_decay.add(full_param_name)
                elif "weight" in param_name:
                    if isinstance(module, whitelist_weight_modules):
                        decay.add(full_param_name)
                    elif isinstance(module, blacklist_weight_modules):
                        no_decay.add(full_param_name)
                elif not ("weight" in param_name or "bias" in param_name):
                    no_decay.add(full_param_name)
        param_dict = {
            param_name: param for param_name, param in self.named_parameters()
        }
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0
        assert len(param_dict.keys() - union_params) == 0

        optim_groups = [
            {
                "params": [
                    param_dict[param_name] for param_name in sorted(list(decay))
                ],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [
                    param_dict[param_name] for param_name in sorted(list(no_decay))
                ],
                "weight_decay": 0.0,
            },
        ]

        # Get optimizer
        optimizer = torch.optim.AdamW(
            optim_groups, lr=self.lr, weight_decay=self.weight_decay
        )

        # Get lr_scheduler
        scheduler = WarmupCosLR(
            optimizer=optimizer,
            lr=self.lr,
            min_lr=1e-6,
            epochs=self.epochs,
            warmup_epochs=self.warmup_epochs,
        )

        # return [optimizer], [scheduler]
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'gradient_clip_val': 3.0,  # Adjust this value to the desired gradient clipping value
        }
