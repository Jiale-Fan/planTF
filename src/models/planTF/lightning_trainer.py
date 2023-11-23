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

logger = logging.getLogger(__name__)


class LightningTrainer(pl.LightningModule):
    def __init__(
        self,
        model: TorchModuleWrapper,
        lr,
        weight_decay,
        epochs,
        warmup_epochs,
        contrastive_weight = 1000.0,
        contrastive_temperature = 0.1,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["model"])

        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.warmup_epochs = warmup_epochs
        self.temperature = contrastive_temperature # TODO: adjust temperature?
        self.contrastive_criterion = nn.CrossEntropyLoss()
        self.neg_num = 2 # TODO: adjust neg_num?
        self.contrastive_weight = contrastive_weight # TODO: adjust contrastive_weight?

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
        # look up tables to get positive and negative pairs TODO

        # pos_pairs, neg_pairs = self.get_positive_negative_pairs(data["scenario_type"])
        # pos_emb = scenario_emb_proj[pos_pairs]
        # neg_emb = torch.stack([scenario_emb_proj[neg_pairs[i]] for i in range(scenario_emb_proj.size(0))], dim=0)
        # contrastive_loss = self._contrastive_loss(scenario_emb_proj, pos_emb, neg_emb)

        # 2. contrastive loss between multi-modal plans
        contrastive_loss = 0
        if self.training:
            contrastive_loss = self._contrastive_loss(res['scene_plan_emb_proj'], res["scene_pos_emb_proj"], res["scene_neg_emb_proj"])

        loss = nll_loss + adefde_loss + kl_loss + self.contrastive_weight*contrastive_loss

        return {
            "loss": loss,
            # "reg_loss": nll_loss,
            # "cls_loss": 0,
            # "prediction_loss": 0,
            "kl_loss": kl_loss,
            "post_entropy": post_entropy,
            "ade_fde_loss": adefde_loss,
            "nll_loss": nll_loss,
            "contrastive_loss": contrastive_loss,
        }
    
    def get_positive_negative_pairs(self, scenario_types_ids):
        '''
        Input:
            scenario_types_ids: [batch_size]
        Output:
            pos_pairs: [batch_size]
            neg_pairs: [batch_size, num_neg]
        '''
        pass # TODO
        b = scenario_types_ids.size(0)
        pos_pairs = torch.zeros(b)
        neg_pairs = torch.zeros(b, self.neg_num)

        # temporary placeholders
        return pos_pairs.to(torch.long), neg_pairs.to(torch.long)

    
    def _contrastive_loss(self, emb_query, emb_pos, emb_neg):
        '''
        Input:
            emb_query: [batch, dim]
            emb_pos: [batch, dim]
            emb_neg: [batch, num_neg, dim]
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

        # loss (social-nce)

        # logits = (torch.cat([sim_pos.unsqueeze(1), sim_neg], dim=1) / self.temperature)
        # labels = torch.zeros(logits.size(0), dtype=torch.long, device=self.device)
        # loss = self.contrastive_criterion(logits, labels)

        # loss
        loss = -torch.log(torch.exp(sim_pos / self.temperature) / torch.sum(torch.exp(sim_neg / self.temperature), dim=-1)).mean()

        return loss


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
