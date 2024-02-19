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

    def unpack(self) -> List[AbstractModelFeature]:
        raise NotImplementedError

    def is_valid(self) -> bool:
        return self.data["polylines"].shape[0] > 0
    
    @classmethod
    def batch_rotation_matmul(self, matrix: np.mat, rotate_mat: np.mat):
        original_shape = matrix.shape
        if len(matrix.shape) == 4:
            b = np.repeat(rotate_mat.transpose(2,0,1)[:, np.newaxis, :, :], matrix.shape[1], axis=1)
        elif len(matrix.shape) == 5:
            b = np.repeat(rotate_mat.transpose(2,0,1)[:, np.newaxis, :, :], matrix.shape[1], axis=1)
            b = np.repeat(b[:, :, np.newaxis, :, :], matrix.shape[2], axis=2)
        flatten_matrix = matrix.reshape(-1, original_shape[-2], original_shape[-1])
        flatten_b = b.reshape(-1, b.shape[-2], b.shape[-1])
        res = np.matmul(flatten_matrix, flatten_b)
        return res.reshape(original_shape)

    @classmethod
    def normalize(
        self, data, first_time=False, radius=None, hist_steps=21, batch = False
    ) -> NuplanFeature:
        cur_state = data["current_state"]
        if batch == False:
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

        else:
            center_xy, center_angle = cur_state[:, :2].copy(), cur_state[:, 2].copy()

            rotate_mat = np.array(
                [
                    [np.cos(center_angle), -np.sin(center_angle)],
                    [np.sin(center_angle), np.cos(center_angle)],
                ],
                dtype=np.float64,
            )

            data["current_state"][:, :3] = 0
            data["agent"]["position"] = NuplanFeature.batch_rotation_matmul(
                data["agent"]["position"] - center_xy[:, None, None, :], rotate_mat
            )
            data["agent"]["velocity"] = NuplanFeature.batch_rotation_matmul(data["agent"]["velocity"], rotate_mat)
            data["agent"]["heading"] -= center_angle[:,None,None]

            data["map"]["point_position"] = NuplanFeature.batch_rotation_matmul(
                data["map"]["point_position"] - center_xy[:, None, None, None, :], rotate_mat
            )
            data["map"]["point_vector"] = NuplanFeature.batch_rotation_matmul(data["map"]["point_vector"], rotate_mat)
            data["map"]["point_orientation"] -= center_angle[:,None,None,None]

            data["map"]["polygon_center"][..., :2] = np.matmul(
                data["map"]["polygon_center"][..., :2] - center_xy[:, None, :], rotate_mat.transpose(2,0,1)
            )
            data["map"]["polygon_center"][..., 2] -= center_angle[:,None]
            data["map"]["polygon_position"] = np.matmul(
                data["map"]["polygon_position"] - center_xy[:, None, :], rotate_mat.transpose(2,0,1)
            )
            data["map"]["polygon_orientation"] -= center_angle[:,None]

            target_position = (
                data["agent"]["position"][:, :, hist_steps:]
                - data["agent"]["position"][:, :, hist_steps - 1][:, :, None]
            )
            target_heading = (
                data["agent"]["heading"][:, :, hist_steps:]
                - data["agent"]["heading"][:, :, hist_steps - 1][:, :, None]
            )
            target = np.concatenate([target_position, target_heading[..., None]], -1)
            target[~data["agent"]["valid_mask"][:, :, hist_steps:]] = 0

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
