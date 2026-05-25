import random

import torch

from pqdt.data.deformed_shapenet import deform_point_cloud_with_gabor_batched
from pqdt.utils.miscs import separate_point_cloud


def prepare_restoration_batch(batch, cfg, stage):
    result = dict(batch)
    if "source_points" in result and "target_points" in result:
        return result

    dataset_name = cfg.dataset.name
    n_points = cfg.dataset.n_points
    target_points = result["target_points"]

    if dataset_name == "shapenet55":
        source_points, _ = separate_point_cloud(
            target_points,
            n_points,
            [int(n_points * 1 / 4), int(n_points * 3 / 4)],
            fixed_points=None,
        )
    elif dataset_name == "shapenet_deform":
        if stage == "train":
            deformed_points = deform_point_cloud_with_gabor_batched(
                target_points,
                scale=random.uniform(0.2, 0.6),
                num_kernels=random.randint(8, 24),
                freq=random.uniform(1.0, 3.0),
                sigma=random.uniform(0.4, 0.6),
                seed=-1,
            )
        else:
            deformed_points = deform_point_cloud_with_gabor_batched(
                target_points,
                scale=0.4,
                num_kernels=16,
                freq=2.0,
                sigma=0.5,
                seed=42,
            )
        source_points, _ = separate_point_cloud(
            deformed_points,
            n_points,
            [int(n_points * 1 / 4), int(n_points * 3 / 4)],
            fixed_points=None,
        )
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    result["source_points"] = source_points
    result["target_points"] = target_points
    return result


def crop_directions(device):
    return [
        torch.tensor([1, 1, 1], device=device, dtype=torch.float32),
        torch.tensor([1, 1, -1], device=device, dtype=torch.float32),
        torch.tensor([1, -1, 1], device=device, dtype=torch.float32),
        torch.tensor([-1, 1, 1], device=device, dtype=torch.float32),
        torch.tensor([-1, -1, 1], device=device, dtype=torch.float32),
        torch.tensor([-1, 1, -1], device=device, dtype=torch.float32),
        torch.tensor([1, -1, -1], device=device, dtype=torch.float32),
        torch.tensor([-1, -1, -1], device=device, dtype=torch.float32),
    ]
