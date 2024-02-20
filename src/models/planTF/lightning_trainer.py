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
from src.models.planTF.training_objectives import nll_loss_multimodes_joint, nll_loss_multimodes
from src.models.planTF.pairing_matrix import proj_name_to_mat
from src.models.planTF.planning_model import get_rotated_data

from torch.nn.utils.rnn import pad_sequence
from src.features.nuplan_feature import NuplanFeature


logger = logging.getLogger(__name__)


class LightningTrainer(pl.LightningModule):
    def __init__(
        self,
        model: TorchModuleWrapper,
        lr,
        weight_decay,
        epochs,
        warmup_epochs,
        pretraining_epochs=0,
        masker_var_weight=1.0,
        prediction_timesteps=40,
        lam = 0.5,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["model"])

        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.warmup_epochs = warmup_epochs

        self.automatic_optimization = False

        self.optimize_term_switch = False
        self.pretraining_epochs = pretraining_epochs
        self.masker_var_weight = masker_var_weight
        self.stage_two_init_flag = False
        self.prediction_timesteps = prediction_timesteps

        self.lam = lam


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
    
    def _plantf_loss(self, targets, trajectory, probability, prediction, valid_mask ):
        ego_target_pos, ego_target_heading = targets[:, 0, :, :2], targets[:, 0, :, 2]
        ego_target = torch.cat(
            [
                ego_target_pos,
                torch.stack(
                    [ego_target_heading.cos(), ego_target_heading.sin()], dim=-1
                ),
            ],
            dim=-1,
        )
        agent_target, agent_mask = targets[:, 1:], valid_mask[:, 1:]

        ade = torch.norm(trajectory[..., :2] - ego_target[:, None, :, :2], dim=-1)
        best_mode = torch.argmin(ade.sum(-1), dim=-1)
        best_traj = trajectory[torch.arange(trajectory.shape[0]), best_mode]
        ego_reg_loss_batch = F.smooth_l1_loss(best_traj, ego_target, reduction="none").mean((-2,-1))
        ego_reg_loss = ego_reg_loss_batch.mean()
        ego_reg_loss_var = ego_reg_loss_batch.var()
        ego_cls_loss = F.cross_entropy(probability, best_mode.detach())

        agent_reg_loss = F.smooth_l1_loss(
            prediction[agent_mask], agent_target[agent_mask][:, :2]
        )

        total_loss = ego_cls_loss + ego_reg_loss + agent_reg_loss

        return total_loss, ego_reg_loss_var


    def _compute_objectives(self, res, data) -> Dict[str, torch.Tensor]:
        trajectory, keyframes, rot_angle = (
            res["trajectory"],
            res["keyframes"],
            res["rot_angle"],
        )        

        targets = data["agent"]["target"]
        valid_mask = data["agent"]["valid_mask"][:, :, -trajectory.shape[-2] :]

        targets_with_angles = torch.cat(
            [
                targets[:, :, :, :2],
                torch.stack(
                    [targets[:, :, :, 2].cos(), targets[:, :, :, 2].sin()], dim=-1
                ),
            ],
            dim=-1,
        )

        # agent_reg_loss = F.smooth_l1_loss(
        #     prediction[valid_mask], targets_with_angles[valid_mask]
        # )

        data_rot = get_rotated_data(data, rot_angle)
        targets_rot = data_rot["agent"]["target"]
        ego_target_pos_rot, ego_target_heading_rot = targets_rot[:, 0, :, :2], targets_rot[:, 0, :, 2]
        ego_target_rot = torch.cat(
            [
                ego_target_pos_rot,
                torch.stack(
                    [ego_target_heading_rot.cos(), ego_target_heading_rot.sin()], dim=-1
                ),
            ],
            dim=-1,
        )

        ego_keyframes_gt = ego_target_rot[:, self.model.keyframes_indices].to(keyframes.device)
        key_frames_loss = F.smooth_l1_loss(keyframes, ego_keyframes_gt, reduction="none").mean(-1) # (B, K)
        
        train_epoch_progress = (self.current_epoch-self.pretraining_epochs) / (self.epochs-self.pretraining_epochs)

        # for debug 
        # train_epoch_progress = 0.3

        keyframes_loss_effective = key_frames_loss[:, key_frames_loss.shape[-1]-int(self.model.num_keyframes*train_epoch_progress):]

        key_frames_ade_loss = keyframes_loss_effective.mean()
        key_frames_fde_loss = key_frames_loss[:, -1].mean()

        if self.training:
            ego_target_pos, ego_target_heading = targets[:, 0, :, :2], targets[:, 0, :, 2]
            ego_target = torch.cat(
                [
                    ego_target_pos,
                    torch.stack(
                        [ego_target_heading.cos(), ego_target_heading.sin()], dim=-1
                    ),
                ],
                dim=-1,
            )
            trajectory_ref_loss = res["trajectory_ref_loss"]
            ego_reg_loss_batch = F.smooth_l1_loss(trajectory_ref_loss, ego_target, reduction="none").mean((-2,-1))
            ego_reg_loss = ego_reg_loss_batch.mean()
            # ego_reg_loss_var = ego_reg_loss_batch.var()
        else:
            ego_reg_loss = 0

        total_loss = ego_reg_loss + key_frames_ade_loss

        return {
            "loss": total_loss,
            "loss_keyframe": key_frames_ade_loss,
            "loss_keyframe_fde": key_frames_fde_loss,
            # "loss_pred": agent_reg_loss,
            "loss_ref": ego_reg_loss,
        }

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


        def training_step(self, batch, batch_idx):
            opt = self.optimizers()
            opt.zero_grad()
            loss = self.compute_loss(batch)
            self.manual_backward(loss)
            opt.step()
        """
        prefix = "train"
        # with torch.autograd.detect_anomaly():

        features, _, _ = batch
        res = self.forward(features["feature"].data)
        losses = self._compute_objectives(res, features["feature"].data)

        opts = self.optimizers()

        opt_main = opts

        # if self.current_epoch < self.pretraining_epochs:
        #     opt_main.zero_grad()
        #     self.manual_backward(losses["loss_pred"]+losses["loss_ref"]) 
        #     opt_main.step()
        # else:
        # # if True:
        #     if not self.stage_two_init_flag:
        #         self.model.init_stage_two()
        #         self.stage_two_init_flag = True
        #         # the init of stage two copies the map encoder for prediction
        #         # to the map encoder for planning. Since we've replaced the original
        #         # parameters, we can not do backward in this step
        #     else:
        #         opt_main.zero_grad()
        #         self.manual_backward(losses["loss"])
        #         opt_main.step()

        opt_main.zero_grad()
        self.manual_backward(losses["loss"])
        opt_main.step()
            
        metrics = self._compute_metrics(res, features["feature"].data, prefix)
        self._log_step(losses["loss"], losses, metrics, prefix) 
        
        return losses["loss"]

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

    def configure_optimizers_memo(
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
    
        # for module_name, module in self.model.get_loss_final_modules():
            for param_name, param in module.named_parameters():
                full_param_name = (
                    "%s.%s" % (module_name, param_name) if module_name else param_name
                )


                # # exclude adv_masker and sup_trajectory_decoder from the main optimizer
                # if "adv_masker" in full_param_name or "sup_trajectory_decoder" in full_param_name:
                #     continue

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

        ## Following assertions are not true because we now use multiple optimizers
        # assert len(inter_params) == 0
        # assert len(param_dict.keys() - union_params) == 0

        decay_params = [
                    param_dict[param_name] for param_name in sorted(list(decay))
                ]
        
        no_decay_params = [
                    param_dict[param_name] for param_name in sorted(list(no_decay))
                ]

        optim_groups_full = [
            {
                "params": decay_params,
                "weight_decay": self.weight_decay,
            },
            {
                "params": no_decay_params,
                "weight_decay": 0.0,
            },
        ]

        # Get optimizer
        optimizer_loss_final = torch.optim.AdamW(
            optim_groups_full, lr=self.lr, weight_decay=self.weight_decay
        )

        # Get lr_scheduler
        scheduler_loss_final = self.get_lr_scheduler(optimizer_loss_final)

        decay_params_per = []
        for param_name in sorted(list(decay)):
            if self.if_deputy_optimizer_param_name(param_name):
                decay_params_per.append(param_dict[param_name])
        no_decay_params_per = []
        for param_name in sorted(list(no_decay)):
            if self.if_deputy_optimizer_param_name(param_name):
                no_decay_params_per.append(param_dict[param_name])

        optim_groups_per = [
            {
                "params": decay_params_per,
                "weight_decay": self.weight_decay,
            },
            {
                "params": no_decay_params_per,
                "weight_decay": 0.0,
            },
        ]


        optimizer_masker = torch.optim.AdamW(
            optim_groups_per
             , lr=self.lr, weight_decay=self.weight_decay
        )

        # optimizer_adv_perturbator = torch.optim.AdamW(
        #     self.model.adv_embedding_offset.parameters(), lr=self.lr, weight_decay=self.weight_decay
        # )

        return [optimizer_loss_final, optimizer_masker], [scheduler_loss_final]
        # return {
        #     'optimizer': [optimizer_loss_final, optimizer_adv_masker, optimizer_sup_decoder],
        #     'lr_scheduler': scheduler_loss_final,
        #     # 'gradient_clip_val': 3.0,  # Adjust this value to the desired gradient clipping value
        # }
    

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

        return [optimizer], [scheduler]

    
    
    def if_deputy_optimizer_param_name(self, param_name):
        # if "noise_distributor" in param_name or "encoder_blocks_latter" in param_name or "trajectory_decoder" in param_name or "agent_predictor" in param_name:
        #     return True
        # else:
        #     return False
        if "trajectory_decoder_per" in param_name:
            return True
        else:
            return False
    
    def get_lr_scheduler(self, optimizer):
        return WarmupCosLR(
            optimizer=optimizer,
            lr=self.lr,
            min_lr=1e-6,
            epochs=self.epochs,
            warmup_epochs=self.warmup_epochs,
        )
