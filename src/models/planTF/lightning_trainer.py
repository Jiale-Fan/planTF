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
from src.models.planTF.planning_model import Stage

logger = logging.getLogger(__name__)


class LightningTrainer(pl.LightningModule):
    def __init__(
        self,
        model: TorchModuleWrapper,
        lr,
        weight_decay,
        epochs,
        warmup_epochs,
        pretrain_epochs = 10,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["model"])

        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.warmup_epochs = warmup_epochs
        self.pretrain_epochs = pretrain_epochs

        self.automatic_optimization=False

        self.initial_finetune_flag = False

    def on_fit_start(self) -> None:
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

        res = self.model(features["feature"].data, self.current_epoch)

        metrics = None

        if 'trajectory' in res and 'probability' in res:
            planning_loss = self._compute_objectives(res, features["feature"].data)
            metrics = self._compute_metrics(res, features["feature"].data, prefix)
            res.update(planning_loss) 

        assert 'loss' in res

        opts = self.optimizers()
        schs = self.lr_schedulers()
        opt_pre, opt_fine = opts

        if self.training:
            if self.model.get_stage(self.current_epoch) == Stage.PRETRAIN_SEP:
                opt_pre.zero_grad()
                self.manual_backward(res["loss"]) 
                self.clip_gradients(opt_pre, gradient_clip_val=5.0, gradient_clip_algorithm="norm")
                opt_pre.step()
                schs[0].step(self.current_epoch)

            elif self.model.get_stage(self.current_epoch) == Stage.PRETRAIN_REPRESENTATION: 
                opt_pre.zero_grad()
                opt_fine.zero_grad()
                self.manual_backward(res["loss"]) 
                self.clip_gradients(opt_pre, gradient_clip_val=5.0, gradient_clip_algorithm="norm")
                self.clip_gradients(opt_fine, gradient_clip_val=5.0, gradient_clip_algorithm="norm")
                opt_pre.step()
                opt_fine.step()
                schs[0].step(self.current_epoch)
                schs[1].step(self.current_epoch)
                
                self.model.EMA_update() # update the teacher model with EMA

            elif self.model.get_stage(self.current_epoch) == Stage.FINETUNE or self.model.get_stage(self.current_epoch) == Stage.ANT_MASK_FINETUNE: 
                opt_pre.zero_grad()
                opt_fine.zero_grad()
                self.manual_backward(res["loss"])
                self.clip_gradients(opt_pre, gradient_clip_val=5.0, gradient_clip_algorithm="norm") 
                self.clip_gradients(opt_fine, gradient_clip_val=5.0, gradient_clip_algorithm="norm") 
                opt_pre.step()
                opt_fine.step()
                schs[0].step(self.current_epoch)
                schs[1].step(self.current_epoch)


        # for sch in self.lr_schedulers():
        #     sch.step(self.current_epoch)

        # TODO: manual gradient clipping?
        logged_loss = {k: v for k, v in res.items() if v.dim() == 0}
        self._log_step(res["loss"], logged_loss, metrics, prefix)
        return res["loss"]

    def _compute_objectives(self, res, data) -> Dict[str, torch.Tensor]:
        trajectory, probability, prediction= (
            res["trajectory"], # [bs, N_mask*n_mode, n_steps, 4]
            res["probability"], # [bs, N_mask*n_mode]
            res["prediction"], # [bs, N_mask, n_agents, n_steps, 2]
        )

        N_mask = prediction.shape[1]
        n_mode = trajectory.shape[1] // N_mask

        targets = data["agent"]["target"]
        valid_mask = data["agent"]["valid_mask"][:, :, -trajectory.shape[-2] :]

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
        # agent_target, agent_mask = targets[:, 1:], valid_mask[:, 1:]

        ade = torch.norm(trajectory[..., :2] - ego_target[:, None, :, :2], dim=-1) # [bs, n_modes, n_steps]
        # best_mode = torch.argmin(ade.sum(-1), dim=-1) # [bs]
        best_mode = torch.argmin(ade[..., -1], dim=-1) # [bs]
        best_traj = trajectory[torch.arange(trajectory.shape[0]), best_mode]
        # best_traj_belongs_to_mask_0 = best_mode < n_mode # [bs]
        # ego_reg_loss = F.smooth_l1_loss(best_traj[best_traj_belongs_to_mask_0], ego_target[best_traj_belongs_to_mask_0], reduction='none').mean((1, 2))
        ego_reg_loss_mean = F.smooth_l1_loss(best_traj[:,-1], ego_target[:,-1], reduction='mean')
        ego_cls_loss = F.cross_entropy(probability, best_mode.detach())

        best_mask_set = best_mode//n_mode
        assert best_mask_set.max() < prediction.shape[1]
        corres_pred = prediction[torch.arange(prediction.shape[0]), best_mask_set]

        agent_target, agent_mask = targets[:, 1:], valid_mask[:, 1:]
        agent_reg_loss = F.smooth_l1_loss(
            corres_pred[agent_mask], agent_target[agent_mask][:, :2]
        )

        # loss = ego_reg_loss_mean + ego_cls_loss + agent_reg_loss
        loss = ego_reg_loss_mean + ego_cls_loss + agent_reg_loss
        
        # if self.current_epoch < self.pretrain_epochs:
        #     loss = res["pretrain_loss"]
        # else:
        #     if not self.initial_finetune_flag:
        #         self.model.initialize_finetune()
        #         print("Initial finetune done")
        #         loss = res["pretrain_loss"]
        #         self.initial_finetune_flag = True
        #     else: 
        #         loss = ego_reg_loss_mean + ego_cls_loss + agent_reg_loss

        return {
            "loss": loss,
            "reg_loss": ego_reg_loss_mean,
            "cls_loss": ego_cls_loss,
            "agent_reg_loss": agent_reg_loss,
            # "pretrain_loss": res["pretrain_loss"],
            # "hist_loss": res["hist_loss"],
            # "future_loss": res["future_loss"],
            # "lane_pred_loss": res["lane_pred_loss"],
            # "hist_rec_pred_loss": res["hist_rec_pred_loss"],
            # "fut_rec_pred_loss": res["fut_rec_pred_loss"],
            # "lane_rec_pred_loss": res["lane_rec_pred_loss"],
            # "hard_ratio": res["hard_ratio"],
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
        """
        
        t = self._step(batch, "train")
        return t

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

    def get_optim_groups(self, modules):
        '''
            modules: List[Tuple[str, nn.Module]]
        '''
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

        for module_name, module in modules:
            for param_name, param in module.named_parameters():
                full_param_name = (
                    "%s.%s" % (module_name, param_name) if module_name else param_name
                )
                if "bias" in param_name:
                    no_decay.add(full_param_name)
                elif "weight" in param_name:
                    # if isinstance(module, whitelist_weight_modules):
                    #     decay.add(full_param_name)
                    # elif isinstance(module, blacklist_weight_modules):
                    #     no_decay.add(full_param_name)
                    if isinstance(module, blacklist_weight_modules):
                        no_decay.add(full_param_name)
                    else:
                        decay.add(full_param_name)
                elif not ("weight" in param_name or "bias" in param_name):
                    no_decay.add(full_param_name)

        param_dict = {
            param_name: param for param_name, param in self.named_parameters()
        }

        # inter_params = decay & no_decay
        # union_params = decay | no_decay

        # assert len(inter_params) == 0
        # assert len(param_dict.keys() - union_params) == 0

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

        return optim_groups

    def configure_optimizers(
        self,
    ) -> Union[Optimizer, Dict[str, Union[Optimizer, _LRScheduler]]]:
        """
        Configures the optimizers and learning schedules for the training.

        :return: optimizer or dictionary of optimizers and schedules
        """
        pretrain_modules_list = self.model.get_pretrain_modules()
        finetune_modules_list = self.model.get_finetune_modules()

        pretrain_modules = []
        finetune_modules = []

        for name, module in self.named_modules():
            if module in pretrain_modules_list:
                pretrain_modules.append((name, module))
            elif module in finetune_modules_list:
                finetune_modules.append((name, module))

        optim_groups_pretrain = self.get_optim_groups(pretrain_modules)
        optim_groups_finetune = self.get_optim_groups(finetune_modules)

        # Get optimizers
        optimizer_pretrain = torch.optim.AdamW(
            optim_groups_pretrain, lr=self.lr, weight_decay=self.weight_decay
        )
        optimizer_finetune_p = torch.optim.AdamW(
            optim_groups_pretrain, lr=self.lr, weight_decay=self.weight_decay
        )
        optimizer_finetune_f = torch.optim.AdamW(
            optim_groups_finetune, lr=self.lr, weight_decay=self.weight_decay
        )

        # Get lr_schedulers
        scheduler_pre = WarmupCosLR(
            optimizer=optimizer_pretrain,
            lr=self.lr,
            min_lr=1e-6,
            starting_epoch=self.model.pretrain_epoch_stages,
            epochs=self.epochs,
            warmup_epochs=self.warmup_epochs,
        )

        scheduler_fine_f = WarmupCosLR(
            optimizer=optimizer_finetune_f,
            lr=self.lr,
            min_lr=1e-6,
            starting_epoch=self.model.pretrain_epoch_stages[1:],
            epochs=self.epochs,
            warmup_epochs=0,
        )

        return [optimizer_pretrain, optimizer_finetune_f], [scheduler_pre, scheduler_fine_f]
