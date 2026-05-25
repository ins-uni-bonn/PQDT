from dataclasses import dataclass

import torch
from torch import nn

from pqdt.models.dgcnn_geo import Transdown
from pqdt.models.pq_blocks import GEEncoder


@dataclass
class PQStemOutput:
    coors: list[torch.Tensor]
    features: list[torch.Tensor]
    coor_c: torch.Tensor
    x1: torch.Tensor


class PQStemEncoder(nn.Module):
    """Reusable PQ point stem: Transdown -> input projection -> first GE encoder."""

    def __init__(
        self,
        in_chans=256,
        embed_dim=384,
        num_heads=6,
        enc_attn=("ge_attn", "attn", "attn", "attn"),
        mlp_ratio=2.0,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        transdown_fps=(512, 128),
        transdown_dims=(64, 256),
        transdown_num_heads=(1, 4),
        transdown_sa_depth=(3, 3),
        transdown_k=(16, 16),
    ):
        super().__init__()
        self.transdown = Transdown(
            in_dim=3,
            fps=list(transdown_fps),
            dims=list(transdown_dims),
            num_heads=list(transdown_num_heads),
            sa_depth=list(transdown_sa_depth),
            k=list(transdown_k),
        )
        self.input_proj = nn.Sequential(
            nn.Conv1d(in_chans, embed_dim, 1),
            nn.BatchNorm1d(embed_dim),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(embed_dim, embed_dim, 1),
        )
        self.encoder_1 = GEEncoder(
            embed_dim,
            num_heads,
            attn_cls=enc_attn,
            mlp_ratio=mlp_ratio,
            drop=drop_rate,
            attn_drop=attn_drop_rate,
        )

    def down(self, points):
        return self.transdown(points)

    def encode_pyramid(self, coors, features):
        coor_c = coors[-1]
        projected = self.input_proj(features[-1]).transpose(1, 2)
        x1 = self.encoder_1(coor_c, projected)
        return PQStemOutput(coors=coors, features=features, coor_c=coor_c, x1=x1)

    def forward(self, points):
        coors, features = self.down(points)
        return self.encode_pyramid(coors, features)

    def forward_dual(self, source_points, sketch_points):
        source_coors, source_features = self.down(source_points)
        sketch_coors, sketch_features = self.down(sketch_points)
        coors = [torch.cat([source_coors[i], sketch_coors[i]], dim=2) for i in range(len(source_coors))]
        features = [
            torch.cat([source_features[i], sketch_features[i]], dim=2)
            for i in range(len(source_features))
        ]
        return self.encode_pyramid(coors, features)
