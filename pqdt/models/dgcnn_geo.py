import torch
from torch import nn
from pointnet2_ops import pointnet2_utils
from pqdt.models.pq_blocks import GEEncoder
from pqdt.utils.miscs import get_knn_index, gather_feature

class Transdown(nn.Module):
    def __init__(self,
                 in_dim=3,
                 fps=[2048, 512, 128],
                 dims=[64, 256, 1024],
                 num_heads=[1, 4, 8],
                 sa_depth=[2, 2, 2],
                 k=[16, 16, 16]
                 ):
        super().__init__()
        self.dims = [8] + dims
        self.input_trans = nn.Conv1d(in_dim, self.dims[0], 1)
        self.down_layers = nn.ModuleList([
            GeoEdgeConv(channels=[self.dims[i]*2, self.dims[i+1]//2, self.dims[i+1]], fps=fps[i], num_heads=num_heads[i], depth=sa_depth[i], k=k[i])
            for i in range(len(fps))
        ])

    def forward(self, coor):
        coor = coor.transpose(1, 2).contiguous()
        f = self.input_trans(coor)
        coors, fs = [coor], [f]
        for layer in self.down_layers:
            coor, f = layer(coor, f)
            coors.append(coor)
            fs.append(f)
        return coors, fs


class GeoEdgeConv(nn.Module):
    def __init__(self, channels, fps, num_heads, depth, mlp_ratio=4., drop_rate=0., k=8):
        super().__init__()
        self.fps = fps
        self.k = k
        self.layer1 = nn.Sequential(nn.Conv2d(channels[0], channels[1], kernel_size=1, bias=False),
                                   nn.GroupNorm(4, channels[1]),
                                   nn.LeakyReLU(negative_slope=0.2)
                                   )

        self.layer2 = nn.Sequential(nn.Conv2d(channels[2], channels[2], kernel_size=1, bias=False),
                                   nn.GroupNorm(4, channels[2]),
                                   nn.LeakyReLU(negative_slope=0.2)
                                   )
        self.attn = GEEncoder(channels[2], num_heads, attn_cls=['attn']*depth, mlp_ratio=mlp_ratio, drop=drop_rate, pos_add=False)

    @staticmethod
    def fps_downsample(coor, x, num_group):
        xyz = coor.transpose(1, 2).contiguous() # b, n, 3
        fps_idx = pointnet2_utils.furthest_point_sample(xyz, num_group)

        combined_x = torch.cat([coor, x], dim=1)

        new_combined_x = (
            pointnet2_utils.gather_operation(
                combined_x, fps_idx
            )
        )

        new_coor = new_combined_x[:, :3]
        new_x = new_combined_x[:, 3:]

        return new_coor, new_x
    

    def forward(self, coor, f): # B 3 N, B C N

        _, group_idx = get_knn_index(coor, k=self.k)
        f = gather_feature(f.transpose(1,2), group_idx).transpose(1,2).transpose(1,3)
        f = self.layer1(f)
        f = f.max(dim=-1, keepdim=False)[0]

        coor_q, f_q = self.fps_downsample(coor, f, self.fps)

        _, group_idx = get_knn_index(coor_q, coor, k=self.k)
        f = gather_feature(f.transpose(1,2), group_idx, f_q.transpose(1,2)).transpose(1,2).transpose(1,3)
        f = self.layer2(f)
        f = f.max(dim=-1, keepdim=False)[0]

        _, group_idx = get_knn_index(coor_q, k=self.k)
        f = f.transpose(1,2)
        f = self.attn(coor_q, f)
        f = f.transpose(1,2)
        coor = coor_q

        return coor, f
