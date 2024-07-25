import torch
import torch.nn as nn

from ..layers.embedding import PointsEncoder
from einops import rearrange
from ..layers.common_layers import build_mlp


class MapEncoder(nn.Module):
    def __init__(
        self,
        dim=128,
    ) -> None:
        super().__init__()

        self.dim = dim
        self.polygon_encoder = PointsEncoder(6, dim)
        self.speed_limit_emb = nn.Sequential(
            nn.Linear(1, dim), nn.ReLU(), nn.Linear(dim, dim)
        )

        self.type_emb = nn.Embedding(3, dim)
        self.on_route_emb = nn.Embedding(2, dim)
        self.traffic_light_emb = nn.Embedding(4, dim)
        self.unknown_speed_emb = nn.Embedding(1, dim)
        self.pos_emb = build_mlp(4, [dim] * 2)

    def forward(self, point_position, polygon_property, valid_mask) -> torch.Tensor:
        """
        Args:
            point_position (Tensor)
            polygon_property (Tensor)
            valid_mask (Tensor)

        Returns:
            torch.Tensor
        """
        # polygon_pos: B M P=20 2
        # polygon_property: B M 5
        B, M = point_position.shape[:2]
        polygon_type, polygon_on_route, polygon_tl_status, polygon_has_speed_limit, polygon_speed_limit = \
            polygon_property.long().unbind(dim=-1)
        polygon_has_speed_limit = polygon_has_speed_limit.bool()
        polygon_start = point_position[:,:,0]

        point_vector = torch.cat([point_position[:, :, 1:] - point_position[:, :, :-1], torch.zeros([B, M, 1, 2], dtype=point_position.dtype, device=point_position.device)], dim=-2)
        point_orientation = torch.atan2(point_vector[:, :, :, 1], point_vector[:, :, :, 0])

        polygon_feature = torch.cat(
            [
                point_position - polygon_start[..., None, :2], # NOTE: now same to planTF, except our vector and orientation are approximatedly calculated from position. Whether including the boundary positions helps?
                point_vector,
                torch.stack(
                    [
                        point_orientation.cos(),
                        point_orientation.sin(),
                    ],
                    dim=-1,
                ),
            ],
            dim=-1,
        )

        bs, M, P, C = polygon_feature.shape
        valid_mask = valid_mask.view(bs * M, P)
        polygon_feature = polygon_feature.reshape(bs * M, P, C)

        x_polygon = self.polygon_encoder(polygon_feature, valid_mask).view(bs, M, -1)

        x_type = self.type_emb(polygon_type)
        x_on_route = self.on_route_emb(polygon_on_route)
        x_tl_status = self.traffic_light_emb(polygon_tl_status)
        x_speed_limit = torch.zeros(bs, M, self.dim, device=x_polygon.device)
        x_speed_limit[polygon_has_speed_limit] = self.speed_limit_emb(
            polygon_speed_limit[polygon_has_speed_limit].float().unsqueeze(-1)
        )
        x_speed_limit[~polygon_has_speed_limit] = self.unknown_speed_emb.weight

        x_polygon += x_type + x_on_route + x_tl_status + x_speed_limit


        pos = torch.cat(
            [polygon_start, torch.stack([point_orientation[..., 0].cos(), point_orientation[..., 0].sin()], dim=-1)], dim=-1
        )
        pos_embed = self.pos_emb(pos)

        return x_polygon, pos_embed
