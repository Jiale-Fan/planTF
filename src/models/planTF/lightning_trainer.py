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
        temperature = 0.7,
        scaling = True,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["model"])

        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.warmup_epochs = warmup_epochs
        self.pretrain_epochs = pretrain_epochs
        self.temperature = temperature
        self.scaling = scaling

        self.rel_weighting_sigma = 8

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
            if res["trajectory"].dim() == 5:
                res = {key: res[key][:, 0] for key in res.keys()}
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
    

    def _cal_ego_loss_term_goal_wp(self, trajectory, probability, goal, waypoints, ego_target):
        ego_goal_target = ego_target[:, -1, :] # [bs, 4]
        ego_waypoints_target = ego_target[:, self.model.waypoints_interval-1::self.model.waypoints_interval, :] # [bs, 8, 4]

        # 1. ego regression coarse to fine loss
        ade = torch.norm(trajectory[..., :2] - ego_target[:, None, :, :2], dim=-1).sum(-1)
        ade = torch.where(ade.isnan(), torch.inf, ade) 
        # !!! this is to prevent nan trajectories triggered by the antagonistic masks.
        # we can do this because if one antagonistic mask is all zeros, the other would be full of ones
        # but this may interfere with identifying nan problems at other stages

        best_mode = torch.argmin(ade, dim=-1) # [bs]
        best_traj = trajectory[torch.arange(trajectory.shape[0]), best_mode]
        ego_reg_loss = F.smooth_l1_loss(best_traj, ego_target, reduction='none').mean((1, 2))
        ego_reg_loss_mean = ego_reg_loss.mean()

        ego_goal_loss = F.smooth_l1_loss(goal, ego_goal_target, reduction='none').mean(-1)
        ego_waypoints_loss = F.smooth_l1_loss(waypoints, ego_waypoints_target, reduction='none').mean((1, 2))
        ego_cls_loss = F.cross_entropy(probability, best_mode.detach(), reduction='none')

        # loss = ego_reg_loss_mean + ego_cls_loss + agent_reg_loss
        # ego_loss = ego_reg_loss_mean + ego_cls_loss + ego_goal_loss + ego_waypoints_loss

        return {
            "reg_loss": ego_reg_loss, # [bs]
            "cls_loss": ego_cls_loss, # [bs]
            "ego_goal_loss": ego_goal_loss, # [bs]
            "ego_waypoints_loss": ego_waypoints_loss, # [bs]
        }
    
    def _cal_ego_loss_term(self, trajectory, probability, ego_target):
        # 1. ego regression coarse to fine loss
        ade = torch.norm(trajectory[..., :2] - ego_target[:, None, :, :2], dim=-1).sum(-1)
        ade = torch.where(ade.isnan(), torch.inf, ade) 
        # !!! this is to prevent nan trajectories triggered by the antagonistic masks.
        # we can do this because if one antagonistic mask is all zeros, the other would be full of ones
        # but this may interfere with identifying nan problems at other stages

        best_mode = torch.argmin(ade, dim=-1) # [bs]
        best_traj = trajectory[torch.arange(trajectory.shape[0]), best_mode]
        ego_reg_loss = F.smooth_l1_loss(best_traj, ego_target, reduction='none').mean((1, 2))
        # ego_reg_loss_mean = ego_reg_loss.mean()

        ego_cls_loss = F.cross_entropy(probability, best_mode.detach(), reduction='none')

        # loss = ego_reg_loss_mean + ego_cls_loss + agent_reg_loss
        # ego_loss = ego_reg_loss_mean + ego_cls_loss + ego_goal_loss + ego_waypoints_loss

        return {
            "reg_loss": ego_reg_loss, # [bs]
            "cls_loss": ego_cls_loss, # [bs]
        }

    def _compute_objectives(self, res, data) -> Dict[str, torch.Tensor]:
        trajectory, probability, prediction= (
            res["trajectory"], # [bs, N_mask*n_mode, n_steps, 4]
            res["probability"], # [bs, N_mask*n_mode]
            res["prediction"], # [bs, n_agents, n_steps, 2]
        )

        waypoints = res["waypoints"] # [bs, 20, 4]
        n_wp = waypoints.shape[1]
        far_future_traj = res["far_future_traj"] # [bs, 60, 4]
        rel_prediction = res["rel_prediction"] # [bs, n_agents-1, n_waypoints, 2]
        
        targets = data["agent"]["target"]
        valid_mask = data["agent"]["valid_mask"][:, :, -prediction.shape[-2] :]

        agent_target, agent_mask = targets[:, 1:], valid_mask[:, 1:]
        # agent_target, agent_mask = targets, valid_mask

        # 1. absolute agent prediction
        deno = agent_mask.sum((1, 2))
        deno[deno == 0] = 1
        agent_reg_loss = F.smooth_l1_loss(
            prediction*(agent_mask[..., None]), (agent_target*(agent_mask[..., None]))[..., :2], reduction='none'
        ).sum((1, 2, 3))/deno

        # agent_reg_loss = F.smooth_l1_loss(
        #     prediction[agent_mask], agent_target[agent_mask][..., :2], reduction='none'
        # ).mean()

        # 2. relative agent prediction
        ego_target_pos, ego_target_heading = targets[:, 0, :, :2], targets[:, 0, :, 2]
        rel_agent_pos_gt = (targets[:, 1:, :n_wp, :2] - ego_target_pos[:, None, :n_wp, :]) # [bs, n_agents-1, n_waypoints, 2]
        loss_rel_agent_unweighted = F.smooth_l1_loss(rel_prediction, rel_agent_pos_gt, reduction='none').mean((-1)) # [bs, n_agents-1, n_waypoints]

        coeff = self._gaussian_coeff_function(rel_agent_pos_gt.norm(dim=-1)) # [bs, n_agents-1, n_waypoints]
        coeff[~agent_mask[:, :, :n_wp]] = 0
        loss_rel_agent = (loss_rel_agent_unweighted * coeff).sum((1, 2)) / (coeff.sum((1, 2))+1e-6)

        # 3. lane intention loss
        if "lane_intention_loss" in res:
        # if True:
            lane_intention_loss = res["lane_intention_loss"]
            lane_intention_dict = {k: res[k] for k in res.keys() if k.startswith("lane_intention")}
        else: 
            lane_intention_loss = torch.zeros_like(loss_rel_agent)
            lane_intention_dict = {}

        # 4. waypoint loss
        ego_target = torch.cat(
            [
                ego_target_pos,
                torch.stack(
                    [ego_target_heading.cos(), ego_target_heading.sin()], dim=-1
                ),
            ],
            dim=-1,
        )
        waypoint_loss = F.smooth_l1_loss(waypoints, ego_target[:, :n_wp], reduction='none').mean((1, 2))

        # 5. far future loss
        far_future_loss = F.smooth_l1_loss(far_future_traj, ego_target[:, n_wp:], reduction='none').mean((1, 2))
        
        # ego_loss_dict = self._cal_ego_loss_term(trajectory, probability, ego_target)
        ret_dict_batch = {
            "agent_reg_loss": agent_reg_loss,
            "rel_agent_pos_loss": loss_rel_agent,
            "lane_intention_loss": lane_intention_loss,
            "waypoint_loss": waypoint_loss,
            "far_future_loss": far_future_loss,
        }
        # loss = torch.mean(torch.stack([ret_dict[key] for key in ret_dict.keys()]))
        loss_mat = torch.stack(list(ret_dict_batch.values()), dim=-1) # [bs, 5]
        if self.scaling == True:
            reg_loss_normed = (ret_dict_batch["waypoint_loss"] - ret_dict_batch["waypoint_loss"].min()) / (ret_dict_batch["waypoint_loss"].max() - ret_dict_batch["waypoint_loss"].min() + 1e-6) # [bs]
            scale = torch.exp(reg_loss_normed/self.temperature).detach().clone() # [bs]
            loss = (loss_mat.mean(-1) * scale).mean()
        else:
            loss = loss_mat.mean()

        ret_dict_mean = {key: value.mean() for key, value in ret_dict_batch.items()}
        ret_dict_mean.update(lane_intention_dict)
        ret_dict_mean["loss"] = loss

        return ret_dict_mean

        # elif trajectory.dim() == 5:
        #     score = res["score"] # [bs, n_element], score ranging from 0 to 1
        #     masks = res["masks"] # [bs, N_mask, n_element], binary masks with False for valid elements

        #     comparison_key = "reg_loss" # the key can be changed
        #     ego_loss_dict_list = [self._cal_ego_loss_term(trajectory[:, i], probability[:, i], goal[:, i], waypoints[:, i], ego_target) for i in range(trajectory.shape[1])]
        #     stacked_tensor_list = {key: torch.stack([ego_loss_dict[key] for ego_loss_dict in ego_loss_dict_list], dim=1) for key in ego_loss_dict_list[0].keys()}
        #     better_mask = torch.argmin(stacked_tensor_list[comparison_key], dim=1) # [bs]
        #     selected_losses = {key: torch.gather(stacked_tensor_list[key], 1, better_mask[:, None]) for key in stacked_tensor_list.keys()}
        #     mean_loss_dict = {key: selected_losses[key].mean() for key in selected_losses.keys()}
        #     mean_loss_dict.update({"agent_reg_loss": agent_reg_loss})

        #     score_loss_1 = F.mse_loss(score, torch.where(better_mask == 0, 1.0, -1.0).unsqueeze(-1).repeat(1, score.shape[-1]), reduction='none')
        #     score_loss_2 = F.mse_loss(score, torch.where(better_mask == 1, 1.0, -1.0).unsqueeze(-1).repeat(1, score.shape[-1]), reduction='none')
        #     score_loss = score_loss_1[~masks[:, 0]].mean() + score_loss_2[~masks[:, 1]].mean()
        #     mean_loss_dict.update({"score_loss": score_loss, "0_better_ratio": (better_mask == 0).float().mean()})

        #     mean_loss_dict.update({"loss": torch.mean(torch.stack([mean_loss_dict[key] for key in mean_loss_dict.keys()]))})
        #     mean_loss_dict.update({"0_better_ratio": (better_mask == 0).float().sum()})
        #     mean_loss_dict.update({"sup_better_ratio": (better_mask == 0).float().mean()})
        #     return mean_loss_dict
        # else: 
        #     raise ValueError("trajectory dim should be 4 or 5")

        

        # return {
        #     "loss": loss,
        #     "reg_loss": ego_reg_loss_mean,
        #     "cls_loss": ego_cls_loss,
        #     "agent_reg_loss": agent_reg_loss,
        #     "ego_goal_loss": ego_goal_loss,
        #     "ego_waypoints_loss": ego_waypoints_loss,
            # "pretrain_loss": res["pretrain_loss"],
            # "hist_loss": res["hist_loss"],
            # "future_loss": res["future_loss"],
            # "lane_pred_loss": res["lane_pred_loss"],
            # "hist_rec_pred_loss": res["hist_rec_pred_loss"],
            # "fut_rec_pred_loss": res["fut_rec_pred_loss"],
            # "lane_rec_pred_loss": res["lane_rec_pred_loss"],
            # "hard_ratio": res["hard_ratio"],
        # }
    
    # def winner_take_all_loss_cal(self, trajectory, targets):
    def _gaussian_coeff_function(self, x):
        return torch.exp(-x**2/(2*self.rel_weighting_sigma)**2).detach()

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
        # optimizer_finetune_p = torch.optim.AdamW(
        #     optim_groups_pretrain, lr=self.lr, weight_decay=self.weight_decay
        # )
        optimizer_finetune_f = torch.optim.AdamW(
            optim_groups_finetune, lr=self.lr, weight_decay=self.weight_decay
        )

        # Get lr_schedulers
        scheduler_pre = WarmupCosLR(
            optimizer=optimizer_pretrain,
            lr=self.lr,
            min_lr=1e-6,
            starting_epoch=self.model.pretrain_epoch_stages[0:1],
            epochs=self.epochs,
            warmup_epochs=self.warmup_epochs,
        )

        # scheduler_fine_f = WarmupCosLR(
        #     optimizer=optimizer_finetune_f,
        #     lr=self.lr,
        #     min_lr=1e-6,
        #     starting_epoch=self.model.pretrain_epoch_stages[1:],
        #     epochs=self.epochs,
        #     warmup_epochs=0,
        # )

        scheduler_fine_f = WarmupCosLR(
            optimizer=optimizer_finetune_f,
            lr=self.lr,
            min_lr=1e-6,
            starting_epoch=self.model.pretrain_epoch_stages[1:2],
            epochs=self.epochs,
            warmup_epochs=self.warmup_epochs,
        )

        return [optimizer_pretrain, optimizer_finetune_f], [scheduler_pre, scheduler_fine_f]
