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
        pretraining_epochs=6,
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
        probability_full, trajectory_full, predictions_full = (
            res["probability"], # [batch, num_modes, num_agents, time_steps, 5]
            res["trajectory"],
            res["predictions"],
        )

        probability_per, trajectory_per, predictions_per = (
            res["probability_per"], # [batch, num_modes, num_agents, time_steps, 5]
            res["trajectory_per"],
            res["predictions_per"],
        )

        targets = data["agent"]["target"]
        valid_mask = data["agent"]["valid_mask"][..., -targets.shape[-2]:]

        nll_loss_full, kl_loss_full, post_entropy_full, adefde_loss_full, var_full = nll_loss_multimodes_joint(predictions_full.permute(1,3,0,2,4), targets.permute(0,2,1,3), probability_full, valid_mask.permute(0,2,1),
                                        entropy_weight=40.0,
                                        kl_weight=20.0,
                                        use_FDEADE_aux_loss=True,
                                        predict_yaw=True)
        
        nll_loss_per, kl_loss_per, post_entropy_per, adefde_loss_per, var_per = nll_loss_multimodes_joint(predictions_per.permute(1,3,0,2,4), targets.permute(0,2,1,3), probability_per, valid_mask.permute(0,2,1),
                                        entropy_weight=40.0,
                                        kl_weight=20.0,
                                        use_FDEADE_aux_loss=True,
                                        predict_yaw=True)
        
        #  nll_loss_per, kl_loss_per, post_entropy_per, adefde_loss_per, var_per = nll_loss_multimodes_joint(trajectory_sup.unsqueeze(2).permute(1,3,0,2,4), targets[:, 0:1].permute(0,2,1,3), probability_sup, valid_mask.permute(0,2,1),
        #                                 entropy_weight=40.0,
        #                                 kl_weight=20.0,
        #                                 use_FDEADE_aux_loss=True,
        #                                 predict_yaw=True)
        

        loss_per = nll_loss_per + adefde_loss_per + kl_loss_per
        loss_full = nll_loss_full + adefde_loss_full + kl_loss_full

        return {
            "loss": loss_full, # this term named exactly as "loss" is used in the log_step function
            "loss_per": loss_per,
            "loss_full": loss_full,
            "var_per": var_per,
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

        if self.current_epoch >=self.pretraining_epochs:
            if self.optimize_term_switch:

                # update the perturbator
                opts[2].zero_grad()
                self.manual_backward(-losses["loss_per"], retain_graph=True)
                opts[2].step()

            else:

                # update the masker 
                opts[1].zero_grad()
                self.manual_backward(losses["loss_per"]+losses["var_per"], retain_graph=True) # TODO: add variance loss
                opts[1].step()

            self.optimize_term_switch = not self.optimize_term_switch

        # update the main model
        opts[0].zero_grad()
        self.manual_backward(losses["loss_full"])
        opts[0].step()

        metrics = self._compute_metrics(res, features["feature"].data, prefix)
        self._log_step(losses["loss"], losses, metrics, prefix) # TODO: write train step and optimizer configuration
        
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
    
        # for module_name, module in self.model.get_loss_final_modules():
            for param_name, param in module.named_parameters():
                full_param_name = (
                    "%s.%s" % (module_name, param_name) if module_name else param_name
                )


                # exclude adv_masker and sup_trajectory_decoder from the main optimizer
                if "adv_masker" in full_param_name or "sup_trajectory_decoder" in full_param_name:
                    continue

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
        optimizer_loss_final = torch.optim.AdamW(
            optim_groups, lr=self.lr, weight_decay=self.weight_decay
        )

        # Get lr_scheduler
        scheduler_loss_final = self.get_lr_scheduler(optimizer_loss_final)

        optimizer_masker = torch.optim.AdamW(
            self.model.masker.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        optimizer_adv_perturbator = torch.optim.AdamW(
            self.model.adv_embedding_offset.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        return [optimizer_loss_final, optimizer_masker, optimizer_adv_perturbator], [scheduler_loss_final]
        # return {
        #     'optimizer': [optimizer_loss_final, optimizer_adv_masker, optimizer_sup_decoder],
        #     'lr_scheduler': scheduler_loss_final,
        #     # 'gradient_clip_val': 3.0,  # Adjust this value to the desired gradient clipping value
        # }
    
    def get_lr_scheduler(self, optimizer):
        return WarmupCosLR(
            optimizer=optimizer,
            lr=self.lr,
            min_lr=1e-6,
            epochs=self.epochs,
            warmup_epochs=self.warmup_epochs,
        )
