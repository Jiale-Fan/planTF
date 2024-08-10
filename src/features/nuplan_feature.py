from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np
import torch
from nuplan.planning.training.preprocessing.features.abstract_model_feature import (
    AbstractModelFeature,
)
from torch.nn.utils.rnn import pad_sequence

from src.utils.conversion import to_device, to_numpy, to_tensor


@dataclass
class NuplanFeature(AbstractModelFeature):
    data: Dict[str, Any]

    @classmethod
    def collate(cls, feature_list: List[NuplanFeature]) -> NuplanFeature:
        batch_data = {}
        for key in ["agent", "map"]:
            batch_data[key] = {
                k: pad_sequence(
                    [f.data[key][k] for f in feature_list], batch_first=True
                )
                for k in feature_list[0].data[key].keys()
            }
        for key in ["current_state", "origin", "angle", "scenario_type"]:
            batch_data[key] = torch.stack([f.data[key] for f in feature_list], dim=0)

        return NuplanFeature(data=batch_data)

    def to_feature_tensor(self) -> NuplanFeature:
        new_data = {}
        for k, v in self.data.items():
            new_data[k] = to_tensor(v)
        return NuplanFeature(data=new_data)

    def to_numpy(self) -> NuplanFeature:
        new_data = {}
        for k, v in self.data.items():
            new_data[k] = to_numpy(v)
        return NuplanFeature(data=new_data)

    def to_device(self, device: torch.device) -> NuplanFeature:
        new_data = {}
        for k, v in self.data.items():
            new_data[k] = to_device(v, device)
        return NuplanFeature(data=new_data)

    def serialize(self) -> Dict[str, Any]:
        return self.data

    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> NuplanFeature:
        return NuplanFeature(data=data)

    def unpack(self) -> List[NuplanFeature]:
        bs = self.data["current_state"].shape[0]
        list_dict = [{} for _ in range(bs)]
        for k, v in self.data.items():
            if isinstance(v, dict):
                for i in range(bs):
                    list_dict[i][k] = {}
                for kk, vv in v.items():
                    for i in range(bs):
                        list_dict[i][k][kk] = vv[i]
            else:
                for i in range(bs):
                    list_dict[i][k] = v[i]

        return [NuplanFeature(data=d) for d in list_dict]

    def is_valid(self) -> bool:
        return self.data["polylines"].shape[0] > 0
    
    @classmethod
    def proximity_filter(
        self, data, radius, hist_steps=21
    ) -> NuplanFeature:
        point_position = data["map"]["point_position"]
        x_max, x_min = radius, -radius
        y_max, y_min = radius, -radius
        valid_mask = (
            (point_position[:, 0, :, 0] < x_max)
            & (point_position[:, 0, :, 0] > x_min)
            & (point_position[:, 0, :, 1] < y_max)
            & (point_position[:, 0, :, 1] > y_min)
        )
        valid_polygon = valid_mask.any(-1)
        data["map"]["valid_mask"] = valid_mask

        for k, v in data["map"].items():
            data["map"][k] = v[valid_polygon]

        return NuplanFeature(data=data)

    @classmethod
    def normalize(
        self, data, first_time=False, radius=None, hist_steps=21
    ) -> NuplanFeature:
        cur_state = data["current_state"]
        center_xy, center_angle = cur_state[:2].copy(), cur_state[2].copy()

        rotate_mat = np.array(
            [
                [np.cos(center_angle), -np.sin(center_angle)],
                [np.sin(center_angle), np.cos(center_angle)],
            ],
            dtype=np.float64,
        )

        data["current_state"][:3] = 0
        data["agent"]["position"] = np.matmul(
            data["agent"]["position"] - center_xy, rotate_mat
        )
        data["agent"]["velocity"] = np.matmul(data["agent"]["velocity"], rotate_mat)
        data["agent"]["heading"] -= center_angle

        data["map"]["point_position"] = np.matmul(
            data["map"]["point_position"] - center_xy, rotate_mat
        )
        data["map"]["point_vector"] = np.matmul(data["map"]["point_vector"], rotate_mat)
        data["map"]["point_orientation"] -= center_angle

        data["map"]["polygon_center"][..., :2] = np.matmul(
            data["map"]["polygon_center"][..., :2] - center_xy, rotate_mat
        )
        data["map"]["polygon_center"][..., 2] -= center_angle
        data["map"]["polygon_position"] = np.matmul(
            data["map"]["polygon_position"] - center_xy, rotate_mat
        )
        data["map"]["polygon_orientation"] -= center_angle

        target_position = (
            data["agent"]["position"][:, hist_steps:]
            - data["agent"]["position"][:, hist_steps - 1][:, None]
        )
        target_heading = (
            data["agent"]["heading"][:, hist_steps:]
            - data["agent"]["heading"][:, hist_steps - 1][:, None]
        )
        target = np.concatenate([target_position, target_heading[..., None]], -1)
        target[~data["agent"]["valid_mask"][:, hist_steps:]] = 0
        data["agent"]["target"] = target

        if first_time:
            point_position = data["map"]["point_position"]
            x_max, x_min = radius, -radius
            y_max, y_min = radius, -radius
            valid_mask = (
                (point_position[:, 0, :, 0] < x_max)
                & (point_position[:, 0, :, 0] > x_min)
                & (point_position[:, 0, :, 1] < y_max)
                & (point_position[:, 0, :, 1] > y_min)
            )
            valid_polygon = valid_mask.any(-1)
            data["map"]["valid_mask"] = valid_mask

            for k, v in data["map"].items():
                data["map"][k] = v[valid_polygon]

            data["origin"] = center_xy
            data["angle"] = center_angle

        return NuplanFeature(data=data)

    @classmethod
    def batch_rotation_matmul(self, matrix, rotate_mat):
        # original_shape = matrix.shape
        # if len(matrix.shape) == 4:
        #     b = torch.repeat(rotate_mat.transpose(2,0,1)[:, np.newaxis, :, :], matrix.shape[1], axis=1)
        # elif len(matrix.shape) == 5:
        #     b = np.repeat(rotate_mat.transpose(2,0,1)[:, np.newaxis, :, :], matrix.shape[1], axis=1)
        #     b = np.repeat(b[:, :, np.newaxis, :, :], matrix.shape[2], axis=2)
        # flatten_matrix = matrix.reshape(-1, original_shape[-2], original_shape[-1])
        # flatten_b = b.reshape(-1, b.shape[-2], b.shape[-1])
        # res = np.matmul(flatten_matrix, flatten_b)
        # return res.reshape(original_shape)
        ori_shape = matrix.shape
        matrix_reshape = matrix.view(ori_shape[0], -1, ori_shape[-1])
        res = torch.matmul(matrix_reshape, rotate_mat)
        return res.view(ori_shape)
    
    @classmethod
    def trajectory_renormalization(self, trajectory, center_xy, rotate_mat):
        # trajectory: [B, num_modes, num_steps, 4(x, y, cos a, sin a)]
        # center_xy: [B, 2]
        # rotate_mat: [B, 2, 2] 
        rotate_mat_inv = rotate_mat.transpose(1, 2)
        new_trajectory_xy = NuplanFeature.batch_rotation_matmul(
            trajectory[..., :2], rotate_mat_inv
        ) + center_xy[:, None, None, :]
        new_trajectory_orientation = NuplanFeature.batch_rotation_matmul(
            trajectory[..., 2:], rotate_mat_inv
        )
        new_trajectory = torch.cat([new_trajectory_xy, new_trajectory_orientation], dim=-1)
        return new_trajectory
    

    @classmethod 
    def time_shift(self, data, shift_steps=5, hist_steps=21) -> dict:

        # make a deep copy of the data first
        new_data = {}
        for key, value in data.items():
            if isinstance(value, torch.Tensor):
                new_data[key] = value.clone()
            elif isinstance(value, dict):
                new_data[key] = {}
                for k, v in value.items():
                    new_data[key][k] = v.clone()
            else:
                new_data[key] = value
            
        assert shift_steps > 0 and shift_steps < hist_steps

        # normalize the data according to its previous state
        center_xy, center_angle = new_data["agent"]["position"][:, 0, hist_steps-shift_steps].clone(), new_data["agent"]["heading"][:, 0, hist_steps-shift_steps].clone()

        rotate_mat = torch.stack(
            [
                torch.cos(center_angle), -torch.sin(center_angle),
                torch.sin(center_angle), torch.cos(center_angle),
            ],
            dim = 1
        ).view(-1, 2, 2) # [bs, 2, 2]

        new_data["current_state"][:, :3] = 0
        new_data["agent"]["position"] = NuplanFeature.batch_rotation_matmul(
            new_data["agent"]["position"] - center_xy[:, None, None, :], rotate_mat
        )
        new_data["agent"]["velocity"] = NuplanFeature.batch_rotation_matmul(new_data["agent"]["velocity"], rotate_mat)
        new_data["agent"]["heading"] -= center_angle[:,None,None]

        new_data["map"]["point_position"] = NuplanFeature.batch_rotation_matmul(
            new_data["map"]["point_position"] - center_xy[:, None, None, None, :], rotate_mat
        )
        new_data["map"]["point_vector"] = NuplanFeature.batch_rotation_matmul(new_data["map"]["point_vector"], rotate_mat)
        new_data["map"]["point_orientation"] -= center_angle[:,None,None,None]

        new_data["map"]["polygon_center"][..., :2] = torch.matmul(
            new_data["map"]["polygon_center"][..., :2] - center_xy[:, None, :], rotate_mat
        )
        new_data["map"]["polygon_center"][..., 2] -= center_angle[:,None]
        new_data["map"]["polygon_position"] = torch.matmul(
            new_data["map"]["polygon_position"] - center_xy[:, None, :], rotate_mat
        )
        new_data["map"]["polygon_orientation"] -= center_angle[:,None]

        # shift the time step for agent data

        for item in ["position", "velocity", "shape"]:
            zeros = torch.zeros(new_data["agent"][item].shape, dtype=new_data["agent"][item].dtype, device=new_data["agent"][item].device)
            new_data["agent"][item] = torch.cat(
                [zeros[:,:,:shift_steps], new_data["agent"][item][:, :, shift_steps:]], dim=-2
            )

        for item in ["heading", "valid_mask"]:
            zeros = torch.zeros(new_data["agent"][item].shape, dtype=new_data["agent"][item].dtype, device=new_data["agent"][item].device)
            new_data["agent"][item] = torch.cat(
                [zeros[:,:,:shift_steps], new_data["agent"][item][:, :, shift_steps:]], dim=-1
            )

        target_position = (
            new_data["agent"]["position"][:, :, hist_steps:]
            - new_data["agent"]["position"][:, :, hist_steps - 1][:, :, None]
        )
        target_heading = (
            new_data["agent"]["heading"][:, :, hist_steps:]
            - new_data["agent"]["heading"][:, :, hist_steps - 1][:, :, None]
        )
        target = torch.concat([target_position, target_heading[..., None]], -1)
        target[~new_data["agent"]["valid_mask"][:, :, hist_steps:]] = 0

        new_data["agent"]["target"] = target

        return new_data, center_xy, rotate_mat