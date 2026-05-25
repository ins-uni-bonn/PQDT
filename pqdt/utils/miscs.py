import numpy as np
import math
import torch
from torch import nn
import torch.nn.functional as F
import random
from torch.nn.init import trunc_normal_
from pointnet2_ops.pointnet2_utils import furthest_point_sample, gather_operation, ball_query, grouping_operation


def knn_point(nsample, xyz, new_xyz):
    """
    Input:
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    sqrdists = square_distance(new_xyz, xyz)
    _, group_idx = torch.topk(sqrdists, nsample, dim = -1, largest=False, sorted=False)
    return group_idx

def get_knn_index(coor_q, coor_k=None, k=8):
    coor_k = coor_k if coor_k is not None else coor_q
    # coor: bs, 3, np
    batch_size, _, num_points = coor_q.size()
    num_points_k = coor_k.size(2)

    with torch.no_grad():
        #idx_b = knn_16(coor_k.transpose(-1, -2).contiguous(), coor_q.transpose(-1, -2).contiguous())[-1]  # bs k np
        idx_b = knn_point(k, coor_k.transpose(-1, -2).contiguous(), coor_q.transpose(-1, -2).contiguous()) # B G M
        idx = idx_b.transpose(-1, -2).contiguous()
        idx_base = torch.arange(0, batch_size, device=coor_q.device).view(-1, 1, 1) * num_points_k
        idx = idx + idx_base
        idx = idx.view(-1)
    
    return idx, idx_b  # bs*k*np, bs np k

def gather_feature(x, group_idx, x_q=None):
    """
    Args:
        x: (B, N_src, C)
        group_idx: (B, N, K)
        x_q (optional): (B, N, C)

    Returns:
        gathered: (B, N, K, 2C)
    """
    B, N_src, C = x.size()
    _, N, K = group_idx.size()

    # Reshape group_idx for gather
    group_idx_flat = group_idx.view(B, -1)  # (B, N*K)

    # Prepare x for gather
    x_flat = x  # (B, N_src, C)

    # Gather features based on group_idx
    gathered = torch.gather(
        x_flat, 1,
        group_idx_flat.unsqueeze(-1).expand(-1, -1, C)
    )  # (B, N*K, C)

    # Reshape to (B, N, K, C)
    gathered = gathered.view(B, N, K, C)

    # Use x_q or x for center features
    if x_q is None:
        x_q = x  # (B, N_src, C)
    x_center = torch.gather(
        x_q, 1,
        group_idx[..., 0].unsqueeze(-1).expand(-1, -1, C)
    ) if x_q.size(1) == N_src else x_q  # (B, N, C)

    x_center = x_center.unsqueeze(2)  # (B, N, 1, C)

    # Broadcast and concat
    output = torch.cat((gathered - x_center, x_center.expand(-1, -1, K, -1)), dim=-1)  # (B, N, K, 2C)
    return output


class SkipFeatureGather(nn.Module):
    def __init__(self, dim, tri=False):
        super(SkipFeatureGather, self).__init__()
        self.tri = tri
        self.map1 = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.LeakyReLU(negative_slope=0.2)
        )
        if tri:
            self.map2 = nn.Sequential(
                nn.Linear(dim * 2, dim),
                nn.LeakyReLU(negative_slope=0.2)
            )
            self.merge_map = nn.Linear(dim*3, dim)
        else:
            self.merge_map = nn.Linear(dim*2, dim)

    def forward(self, prev_f, f, group_idx, f_q=None):
        prev_f_g = gather_feature(prev_f, group_idx, f_q)
        prev_f_g = self.map1(prev_f_g).max(dim=2, keepdim=False)[0]
        if self.tri:
            f_g = gather_feature(f, group_idx)
            f_g = self.map2(f_g).max(dim=2, keepdim=False)[0]
            f = torch.cat([f, prev_f_g, f_g], dim=-1)
        else:
            f = torch.cat([f, prev_f_g], dim=-1)
        f = self.merge_map(f)
        return f
    

class Conv1d(nn.Module):
    def __init__(self,
                 in_channel,
                 out_channel,
                 kernel_size=1,
                 stride=1,
                 if_bn=True,
                 activation_fn=torch.relu):
        super(Conv1d, self).__init__()
        self.conv = nn.Conv1d(in_channel,
                              out_channel,
                              kernel_size,
                              stride=stride)
        self.if_bn = if_bn
        self.bn = nn.BatchNorm1d(out_channel)
        self.activation_fn = activation_fn

    def forward(self, input):
        out = self.conv(input)
        if self.if_bn:
            out = self.bn(out)

        if self.activation_fn is not None:
            out = self.activation_fn(out)

        return out


class Conv2d(nn.Module):
    def __init__(self,
                 in_channel,
                 out_channel,
                 kernel_size=(1, 1),
                 stride=(1, 1),
                 if_bn=True,
                 activation_fn=torch.relu):
        super(Conv2d, self).__init__()
        self.conv = nn.Conv2d(in_channel,
                              out_channel,
                              kernel_size,
                              stride=stride)
        self.if_bn = if_bn
        self.bn = nn.BatchNorm2d(out_channel)
        self.activation_fn = activation_fn

    def forward(self, input):
        out = self.conv(input)
        if self.if_bn:
            out = self.bn(out)

        if self.activation_fn is not None:
            out = self.activation_fn(out)

        return out


class MLP(nn.Module):
    def __init__(self, in_channel, layer_dims, bn=None, init_weights=False):
        super(MLP, self).__init__()
        layers = []
        last_channel = in_channel
        for out_channel in layer_dims[:-1]:
            layers.append(nn.Linear(last_channel, out_channel))
            if bn:
                layers.append(nn.BatchNorm1d(out_channel))
            layers.append(nn.ReLU())
            last_channel = out_channel
        layers.append(nn.Linear(last_channel, layer_dims[-1]))
        self.mlp = nn.Sequential(*layers)
        if init_weights:
            self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight.data, 1)
            nn.init.constant_(m.bias.data, 0)

    def forward(self, inputs):
        return self.mlp(inputs)


class MLP_CONV(nn.Module):
    def __init__(self, in_channel, layer_dims, bn=None, init_weights=False):
        super(MLP_CONV, self).__init__()
        layers = []
        last_channel = in_channel
        for out_channel in layer_dims[:-1]:
            layers.append(nn.Conv1d(last_channel, out_channel, 1))
            if bn:
                layers.append(nn.BatchNorm1d(out_channel))
            layers.append(nn.ReLU())
            last_channel = out_channel
        layers.append(nn.Conv1d(last_channel, layer_dims[-1], 1))
        self.mlp = nn.Sequential(*layers)
        if init_weights:
            self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv1d):
            nn.init.xavier_normal_(m.weight.data, gain=1)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight.data, 1)
            nn.init.constant_(m.bias.data, 0)

    def forward(self, inputs):
        return self.mlp(inputs)


class MLP_Res(nn.Module):
    def __init__(self, in_dim=128, hidden_dim=None, out_dim=128):
        super(MLP_Res, self).__init__()
        if hidden_dim is None:
            hidden_dim = in_dim
        self.conv_1 = nn.Conv1d(in_dim, hidden_dim, 1)
        self.conv_2 = nn.Conv1d(hidden_dim, out_dim, 1)
        self.conv_shortcut = nn.Conv1d(in_dim, out_dim, 1)

    def forward(self, x):
        """
        Args:
            x: (B, out_dim, n)
        """
        shortcut = self.conv_shortcut(x)
        out = self.conv_2(torch.relu(self.conv_1(x))) + shortcut
        return out


def sample_and_group(xyz, points, npoint, nsample, radius, use_xyz=True):
    """
    Args:
        xyz: Tensor, (B, 3, N)
        points: Tensor, (B, f, N)
        npoint: int
        nsample: int
        radius: float
        use_xyz: boolean

    Returns:
        new_xyz: Tensor, (B, 3, npoint)
        new_points: Tensor, (B, 3 | f+3 | f, npoint, nsample)
        idx_local: Tensor, (B, npoint, nsample)
        grouped_xyz: Tensor, (B, 3, npoint, nsample)

    """
    xyz_flipped = xyz.permute(0, 2, 1).contiguous()  # (B, N, 3)
    new_xyz = gather_operation(xyz,
                               furthest_point_sample(xyz_flipped,
                                                     npoint))  # (B, 3, npoint)

    idx = ball_query(radius, nsample, xyz_flipped,
                     new_xyz.permute(0, 2,
                                     1).contiguous())  # (B, npoint, nsample)
    grouped_xyz = grouping_operation(xyz, idx)  # (B, 3, npoint, nsample)
    grouped_xyz -= new_xyz.unsqueeze(3).repeat(1, 1, 1, nsample)

    if points is not None:
        grouped_points = grouping_operation(points,
                                            idx)  # (B, f, npoint, nsample)
        if use_xyz:
            new_points = torch.cat([grouped_xyz, grouped_points], 1)
        else:
            new_points = grouped_points
    else:
        new_points = grouped_xyz

    return new_xyz, new_points, idx, grouped_xyz


def sample_and_group_all(xyz, points, use_xyz=True):
    """
    Args:
        xyz: Tensor, (B, 3, nsample)
        points: Tensor, (B, f, nsample)
        use_xyz: boolean

    Returns:
        new_xyz: Tensor, (B, 3, 1)
        new_points: Tensor, (B, f|f+3|3, 1, nsample)
        idx: Tensor, (B, 1, nsample)
        grouped_xyz: Tensor, (B, 3, 1, nsample)
    """
    b, _, nsample = xyz.shape
    device = xyz.device
    new_xyz = torch.zeros((1, 3, 1), dtype=torch.float,
                          device=device).repeat(b, 1, 1)
    grouped_xyz = xyz.reshape((b, 3, 1, nsample))
    idx = torch.arange(nsample,
                       device=device).reshape(1, 1, nsample).repeat(b, 1, 1)
    if points is not None:
        if use_xyz:
            new_points = torch.cat([xyz, points], 1)
        else:
            new_points = points
        new_points = new_points.unsqueeze(2)
    else:
        new_points = grouped_xyz

    return new_xyz, new_points, idx, grouped_xyz


def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.

    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))  # B, N, M
    dist += torch.sum(src**2, -1).view(B, N, 1)
    dist += torch.sum(dst**2, -1).view(B, 1, M)
    return dist


def query_knn(nsample, xyz, new_xyz, include_self=True):
    """Find k-NN of new_xyz in xyz"""
    pad = 0 if include_self else 1
    sqrdists = square_distance(new_xyz, xyz)  # B, S, N
    idx = torch.argsort(sqrdists, dim=-1, descending=False)[:, :,
                                                            pad:nsample + pad]
    return idx.int()


def sample_and_group_knn(xyz, points, npoint, k, use_xyz=True, idx=None):
    """
    Args:
        xyz: Tensor, (B, 3, N)
        points: Tensor, (B, f, N)
        npoint: int
        nsample: int
        radius: float
        use_xyz: boolean

    Returns:
        new_xyz: Tensor, (B, 3, npoint)
        new_points: Tensor, (B, 3 | f+3 | f, npoint, nsample)
        idx_local: Tensor, (B, npoint, nsample)
        grouped_xyz: Tensor, (B, 3, npoint, nsample)

    """
    xyz_flipped = xyz.permute(0, 2, 1).contiguous()  # (B, N, 3)
    new_xyz = gather_operation(xyz,
                               furthest_point_sample(xyz_flipped,
                                                     npoint))  # (B, 3, npoint)
    if idx is None:
        idx = query_knn(k, xyz_flipped, new_xyz.permute(0, 2, 1).contiguous())
    grouped_xyz = grouping_operation(xyz, idx)  # (B, 3, npoint, nsample)
    grouped_xyz -= new_xyz.unsqueeze(3).repeat(1, 1, 1, k)

    if points is not None:
        grouped_points = grouping_operation(points,
                                            idx)  # (B, f, npoint, nsample)
        if use_xyz:
            new_points = torch.cat([grouped_xyz, grouped_points], 1)
        else:
            new_points = grouped_points
    else:
        new_points = grouped_xyz

    return new_xyz, new_points, idx, grouped_xyz


def separate_point_cloud(xyz, num_points, crop, fixed_points=None, padding_zeros=False):
    """Generate an incomplete point cloud by removing points around a crop center."""
    _, n, c = xyz.shape

    assert n == num_points
    assert c == 3
    if crop == num_points:
        return xyz, None

    inputs = []
    crops = []
    for points in xyz:
        if isinstance(crop, list):
            num_crop = random.randint(crop[0], crop[1])
        else:
            num_crop = crop

        points = points.unsqueeze(0)

        if fixed_points is None:
            center = F.normalize(torch.randn(1, 1, 3, device=xyz.device), p=2, dim=-1)
        else:
            if isinstance(fixed_points, list):
                fixed_point = random.sample(fixed_points, 1)[0]
            else:
                fixed_point = fixed_points
            center = fixed_point.reshape(1, 1, 3).to(device=xyz.device, dtype=xyz.dtype)

        distance_matrix = torch.norm(center.unsqueeze(2) - points.unsqueeze(1), p=2, dim=-1)

        idx = torch.argsort(distance_matrix, dim=-1, descending=False)[0, 0]

        if padding_zeros:
            input_data = points.clone()
            input_data[0, idx[:num_crop]] = input_data[0, idx[:num_crop]] * 0
        else:
            input_data = points.clone()[0, idx[num_crop:]].unsqueeze(0)

        crop_data = points.clone()[0, idx[:num_crop]].unsqueeze(0)

        if isinstance(crop, list):
            inputs.append(fps_subsample(input_data, 2048))
            crops.append(torch.zeros_like(inputs[-1]))
        else:
            inputs.append(input_data)
            crops.append(crop_data)

    input_data = torch.cat(inputs, dim=0)
    crop_data = torch.cat(crops, dim=0)

    return input_data.contiguous(), crop_data.contiguous()



def get_nearest_index(target, source, k=1, return_dis=False):
    """
    Args:
        target: (bs, 3, v1)
        source: (bs, 3, v2)
    Return:
        nearest_index: (bs, v1, 1)
    """
    inner = torch.bmm(target.transpose(1, 2), source)  # (bs, v1, v2)
    s_norm_2 = torch.sum(source**2, dim=1)  # (bs, v2)
    t_norm_2 = torch.sum(target**2, dim=1)  # (bs, v1)
    d_norm_2 = s_norm_2.unsqueeze(1) + t_norm_2.unsqueeze(
        2) - 2 * inner  # (bs, v1, v2)
    nearest_dis, nearest_index = torch.topk(d_norm_2,
                                            k=k,
                                            dim=-1,
                                            largest=False)
    if not return_dis:
        return nearest_index
    else:
        return nearest_index, nearest_dis


def indexing_neighbor(x, index):
    """
    Args:
        x: (bs, dim, num_points0)
        index: (bs, num_points, k)
    Return:
        feature: (bs, dim, num_points, k)
    """
    batch_size, num_points, k = index.size()

    id_0 = torch.arange(batch_size).view(-1, 1, 1)

    x = x.transpose(2, 1).contiguous()  # (bs, num_points, num_dims)
    feature = x[id_0, index]  # (bs, num_points, k, num_dims)
    feature = feature.permute(0, 3, 1,
                              2).contiguous()  # (bs, num_dims, num_points, k)

    return feature

def indexing_neighbor_gather(x, index):
    """
    x: (B, C, N0)
    index: (B, N, K)
    Returns:
        feature: (B, C, N, K)
    """
    B, C, N0 = x.shape
    _, N, K = index.shape

    # Flatten x for gather
    x = x.transpose(1, 2).contiguous()  # (B, N0, C)

    # Flatten index to 1D
    index_flat = index.reshape(B, -1)  # (B, N*K)

    # Gather features
    gathered = torch.gather(x, 1, index_flat.unsqueeze(-1).expand(-1, -1, C))  # (B, N*K, C)

    # Reshape to (B, C, N, K)
    gathered = gathered.reshape(B, N, K, C).permute(0, 3, 1, 2).contiguous()

    return gathered


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class GumbelTopK(nn.Module):
    """
    Subset sampler using perturb-and-topk:
        scores = logits + Gumbel(0,1)
        idx = topk(scores, k)

    Returns:
        idx: (B, k) LongTensor with selected indices
    """
    def __init__(self, k: int, noise_scale: float = 1.0):
        super().__init__()
        assert k >= 1
        self.k = k
        self.noise_scale = noise_scale

    def forward(self, logits: torch.Tensor):
        """
        logits: (B, N) tensor
        """
        assert logits.dim() == 2, "logits must be (B, N)"
        B, N = logits.shape
        device, dtype = logits.device, logits.dtype

        # Add Gumbel noise once
        g = _sample_gumbel((B, N), device=device, dtype=dtype)
        scores = logits + self.noise_scale * g

        # Top-k indices
        _, idx = scores.topk(self.k, dim=-1)
        return idx
    

def sample_sphere(num_points: int, batch_size: int, scale: float = 1.0, device="cpu"):
    """
    Uniformly sample random points on the unit sphere (B, N, 3).
    """
    x = torch.randn(batch_size, num_points, 3, device=device)
    x = F.normalize(x, dim=-1)  # normalize each to unit sphere
    return x * scale  # (B, num_points, 3)

def _sample_gumbel(shape, device=None, dtype=None, eps: float = 1e-20):
    """Sample i.i.d. standard Gumbel noise."""
    u = torch.rand(shape, device=device, dtype=dtype).clamp_(min=eps, max=1 - eps)
    return -torch.log(-torch.log(u + eps) + eps)

def get_tau(
    epoch: int, 
    tau0: float = 0.7, 
    tau_min: float = 0.1, 
    total_epochs: int = 200, 
    mode: str = 'linear'
    ) -> float:
    """
    Scheduling function for tau with linear or cosine decay.
    """
    if mode == 'linear':
        return max(tau_min, tau0 * (1.0 - float(epoch) / float(total_epochs)))
    elif mode == 'cosine':
        cos_inner = math.pi * epoch / total_epochs
        return tau_min + 0.5 * (tau0 - tau_min) * (1 + math.cos(cos_inner)) if epoch <= total_epochs else tau_min
    else:
        raise NotImplementedError(f"Unknown mode: {mode}")
    
def fps_subsample(pcd, n_points=2048):
    """
    Args
        pcd: (b, 16384, 3)

    returns
        new_pcd: (b, n_points, 3)
    """
    if pcd.shape[1] == n_points:
        return pcd
    elif pcd.shape[1] < n_points:
        raise ValueError(
            'FPS subsampling receives a larger n_points: {:d} > {:d}'.format(
                n_points, pcd.shape[1]))
    new_pcd = gather_operation(
        pcd.permute(0, 2, 1).contiguous(),
        furthest_point_sample(pcd, n_points))
    new_pcd = new_pcd.permute(0, 2, 1).contiguous()
    return new_pcd

def cat_vertices_color(gts, recons, n_upsample=None):
    """
    Concatinate the first batch of gt pointcloud and reconstructed one with color tensor

    :param gts: ground truth pointclouds with shape (B, 3, N)
    :param recons: reconstructed pointclouds with shape (B, 3, N)
    :return: vis (1, total_points, 3), color (1, total_points, 3)
    """
    gt = gts.permute(0, 2, 1)[0].unsqueeze(0)
    recon = recons.permute(0, 2, 1)[0].unsqueeze(0)
    vis = torch.cat((gt, recon), dim=1)
    gt_color = torch.tensor(np.tile(np.array([0, 255, 0]), (1, gt.shape[1], 1)))
    if n_upsample is None:
        recon_color = torch.tensor(np.tile(np.array([0, 0, 255]), (1, recon.shape[1], 1)))
    else:
        group_size = int(np.prod(n_upsample)) if isinstance(n_upsample, (list, tuple)) else int(n_upsample)
        if group_size <= 0:
            raise ValueError(f"n_upsample must be positive, got {n_upsample}")

        n_groups = math.ceil(recon.shape[1] / group_size)
        recon_color = torch.randint(0, 255, (1, n_groups, 3), dtype=gt_color.dtype)
        recon_color = recon_color.repeat_interleave(group_size, dim=1)[:, : recon.shape[1], :]

    color = torch.cat((gt_color, recon_color), dim=1)
    return vis, color
    
    
    
