import torch

import chamfer


class ChamferFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, xyz1, xyz2):
        dist1, dist2, idx1, idx2 = chamfer.forward(xyz1, xyz2)
        ctx.save_for_backward(xyz1, xyz2, idx1, idx2)

        return dist1, dist2, idx1, idx2

    @staticmethod
    def backward(ctx, grad_dist1, grad_dist2, grad_idx1, grad_idx2):
        xyz1, xyz2, idx1, idx2 = ctx.saved_tensors
        grad_xyz1, grad_xyz2 = chamfer.backward(xyz1, xyz2, idx1, idx2, grad_dist1, grad_dist2)
        return grad_xyz1, grad_xyz2


class ChamferDistanceL2(torch.nn.Module):
    f''' Chamder Distance L2
    '''
    def __init__(self, ignore_zeros=False):
        super().__init__()
        self.ignore_zeros = ignore_zeros

    def forward(self, xyz1, xyz2):
        batch_size = xyz1.size(0)
        if batch_size == 1 and self.ignore_zeros:
            non_zeros1 = torch.sum(xyz1, dim=2).ne(0)
            non_zeros2 = torch.sum(xyz2, dim=2).ne(0)
            xyz1 = xyz1[non_zeros1].unsqueeze(dim=0)
            xyz2 = xyz2[non_zeros2].unsqueeze(dim=0)

        dist1, dist2, _, _ = ChamferFunction.apply(xyz1, xyz2)
        return torch.mean(dist1) + torch.mean(dist2)

class ChamferDistanceL2_split(torch.nn.Module):
    f''' Chamder Distance L2
    '''
    def __init__(self, ignore_zeros=False):
        super().__init__()
        self.ignore_zeros = ignore_zeros

    def forward(self, xyz1, xyz2):
        batch_size = xyz1.size(0)
        if batch_size == 1 and self.ignore_zeros:
            non_zeros1 = torch.sum(xyz1, dim=2).ne(0)
            non_zeros2 = torch.sum(xyz2, dim=2).ne(0)
            xyz1 = xyz1[non_zeros1].unsqueeze(dim=0)
            xyz2 = xyz2[non_zeros2].unsqueeze(dim=0)

        dist1, dist2, _ ,_ = ChamferFunction.apply(xyz1, xyz2)
        return torch.mean(dist1), torch.mean(dist2)

class ChamferDistanceL1(torch.nn.Module):
    f''' Chamder Distance L1
    '''
    def __init__(self, ignore_zeros=False, hyper=False):
        super().__init__()
        self.ignore_zeros = ignore_zeros
        self.hyper = hyper


    def forward(self, xyz1, xyz2):
        batch_size = xyz1.size(0)
        if batch_size == 1 and self.ignore_zeros:
            non_zeros1 = torch.sum(xyz1, dim=2).ne(0)
            non_zeros2 = torch.sum(xyz2, dim=2).ne(0)
            xyz1 = xyz1[non_zeros1].unsqueeze(dim=0)
            xyz2 = xyz2[non_zeros2].unsqueeze(dim=0)

        dist1, dist2, _, _ = ChamferFunction.apply(xyz1, xyz2)
        # import pdb
        # pdb.set_trace()
        if self.hyper:
            dist1 = arcosh(1 + dist1)
            dist2 = arcosh(1 + dist2)
        else:
            dist1 = torch.sqrt(dist1)
            dist2 = torch.sqrt(dist2)
        return (torch.mean(dist1) + torch.mean(dist2))/2

class ChamferDistanceL1_PM(torch.nn.Module):
    f''' Chamder Distance L1
    '''
    def __init__(self, ignore_zeros=False):
        super().__init__()
        self.ignore_zeros = ignore_zeros

    def forward(self, xyz1, xyz2):
        batch_size = xyz1.size(0)
        if batch_size == 1 and self.ignore_zeros:
            non_zeros1 = torch.sum(xyz1, dim=2).ne(0)
            non_zeros2 = torch.sum(xyz2, dim=2).ne(0)
            xyz1 = xyz1[non_zeros1].unsqueeze(dim=0)
            xyz2 = xyz2[non_zeros2].unsqueeze(dim=0)

        dist1, _, _, _ = ChamferFunction.apply(xyz1, xyz2)
        dist1 = torch.sqrt(dist1)
        return torch.mean(dist1)

def arcosh(x, eps=1e-7):  # pragma: no cover
        # x = x.clamp(-1 + eps, 1 - eps)
        # x = x.clamp(1,)
    
        x = torch.clamp(x, min=1 + eps)
        return torch.log(x + torch.sqrt(1 + x) * torch.sqrt(x - 1))

class DensityAwareChamferDistanceL1(torch.nn.Module):

    def __init__(self, alpha=1000, n_lambda=1, non_reg=False):
        super().__init__()
        self.alpha = alpha
        self.n_lambda = n_lambda
        self.non_reg = non_reg
        
    def forward(self, x, gt):
        x = x.float()
        gt = gt.float()
        batch_size, n_x, _ = x.shape
        batch_size, n_gt, _ = gt.shape
        assert x.shape[0] == gt.shape[0]

        if self.non_reg:
            frac_12 = max(1, n_x / n_gt)
            frac_21 = max(1, n_gt / n_x)
        else:
            frac_12 = n_x / n_gt
            frac_21 = n_gt / n_x

        dist1, dist2, idx1, idx2 = ChamferFunction.apply(x, gt)
        # dist1 (batch_size, n_gt): a gt point finds its nearest neighbour x' in x;
        # idx1  (batch_size, n_gt): the idx of x' \in [0, n_x-1]
        # dist2 and idx2: vice versa
        exp_dist1, exp_dist2 = torch.exp(-dist1 * self.alpha), torch.exp(-dist2 * self.alpha)

        loss1 = []
        loss2 = []
        for b in range(batch_size):
            count1 = torch.bincount(idx1[b])
            weight1 = count1[idx1[b].long()].float().detach() ** self.n_lambda
            weight1 = (weight1 + 1e-6) ** (-1) * frac_21
            loss1.append((- exp_dist1[b] * weight1 + 1.).mean())

            count2 = torch.bincount(idx2[b])
            weight2 = count2[idx2[b].long()].float().detach() ** self.n_lambda
            weight2 = (weight2 + 1e-6) ** (-1) * frac_12
            loss2.append((- exp_dist2[b] * weight2 + 1.).mean())

        loss1 = torch.stack(loss1)
        loss2 = torch.stack(loss2)

        return (torch.mean(loss1) + torch.mean(loss2)) / 2
    

class ChamferDistanceL1Norm(torch.nn.Module):
    f''' Chamder Distance L1 normalized
    '''
    def __init__(self, alpha=0.05, ignore_zeros=False):
        super().__init__()
        self.ignore_zeros = ignore_zeros
        self.alpha = alpha


    def forward(self, xyz1, xyz2, xyz3):
        batch_size = xyz1.size(0)
        if batch_size == 1 and self.ignore_zeros:
            non_zeros1 = torch.sum(xyz1, dim=2).ne(0)
            non_zeros2 = torch.sum(xyz2, dim=2).ne(0)
            non_zeros3 = torch.sum(xyz3, dim=2).ne(0)
            xyz1 = xyz1[non_zeros1].unsqueeze(dim=0)
            xyz2 = xyz2[non_zeros2].unsqueeze(dim=0)
            xyz3 = xyz3[non_zeros3].unsqueeze(dim=0)

        dist12, dist21, _, _ = ChamferFunction.apply(xyz1, xyz2)
        dist23, dist32, _, _ = ChamferFunction.apply(xyz2, xyz3)
        # import pdb
        # pdb.set_trace()
        dist12 = torch.sqrt(dist12)
        dist21 = torch.sqrt(dist21)
        dist23 = torch.sqrt(dist23)
        dist32 = torch.sqrt(dist32)
        cd_12 = torch.mean(dist12, 1) + torch.mean(dist21, 1)
        cd_23 = torch.mean(dist23, 1) + torch.mean(dist32, 1)
        cd_norm = self.alpha * cd_12 / (cd_23 + 1e-8)
        return torch.mean(cd_norm)
    

class ChamferDistanceL1NormNew(torch.nn.Module):
    f''' Chamder Distance L1 normalized
    '''
    def __init__(self, alpha=0.05, ignore_zeros=False):
        super().__init__()
        self.ignore_zeros = ignore_zeros
        self.alpha = alpha


    def forward(self, xyz1, xyz2, xyz3, xyz4):
        # xyz1: pred, xyz2: gt, xyz3: src, xyz4: skt
        batch_size = xyz1.size(0)
        if batch_size == 1 and self.ignore_zeros:
            non_zeros1 = torch.sum(xyz1, dim=2).ne(0)
            non_zeros2 = torch.sum(xyz2, dim=2).ne(0)
            non_zeros3 = torch.sum(xyz3, dim=2).ne(0)
            non_zeros4 = torch.sum(xyz4, dim=2).ne(0)
            xyz1 = xyz1[non_zeros1].unsqueeze(dim=0)
            xyz2 = xyz2[non_zeros2].unsqueeze(dim=0)
            xyz3 = xyz3[non_zeros3].unsqueeze(dim=0)
            xyz4 = xyz4[non_zeros4].unsqueeze(dim=0)

        dist12, dist21, _, _ = ChamferFunction.apply(xyz1, xyz2)
        dist34, dist43, _, _ = ChamferFunction.apply(xyz3, xyz4)
        # import pdb
        # pdb.set_trace()
        dist12 = torch.sqrt(dist12)
        dist21 = torch.sqrt(dist21)
        dist34 = torch.sqrt(dist34)
        dist43 = torch.sqrt(dist43)
        cd_12 = torch.mean(dist12, 1) + torch.mean(dist21, 1)
        cd_43 = torch.mean(dist43, 1)
        cd_norm = self.alpha * cd_12 / (cd_43 + 1e-8)
        return torch.mean(cd_norm)