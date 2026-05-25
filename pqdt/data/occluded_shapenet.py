import logging
import random

import numpy as np
import torch
import torch.utils.data as data
from easydict import EasyDict as edict

from pqdt.data import data_transforms as transforms
from pqdt.data.io import IO
from pqdt.data.paths import occlusion_gt_path, occlusion_partial_points_path, occlusion_split_dir
from pqdt.data.shapenet55 import CAR_TAXONOMY_ID


LOGGER = logging.getLogger(__name__)


def _as_cfg(cfg):
    return edict(cfg) if isinstance(cfg, dict) else cfg


class OccludedShapeNetDataset(data.Dataset):
    """ShapeNet-Car occlusion dataset with stored partial inputs and ground truth."""

    def __init__(self, cfg, split):
        super().__init__()
        cfg = _as_cfg(cfg)
        dataset_cfg = cfg.dataset
        self.data_root = occlusion_split_dir(dataset_cfg)
        self.partial_points_path = occlusion_partial_points_path(dataset_cfg)
        self.gt_path = occlusion_gt_path(dataset_cfg)
        self.npoints = int(dataset_cfg.n_points)
        self.split = split
        self.cars_only = bool(getattr(dataset_cfg, "cars_only", False))
        n_renderings = int(getattr(dataset_cfg, "n_renderings", 1))
        self.n_renderings = n_renderings if self.split == "train" else 1
        self.data_list_file = self.data_root / f"{self.split}.txt"

        LOGGER.info("Opening ShapeNet-Occ split file: %s", self.data_list_file)
        with self.data_list_file.open("r") as f:
            lines = [line.strip() for line in f if line.strip()]

        self.file_list = [sample for sample in (self._parse_line(line) for line in lines) if sample is not None]
        LOGGER.info("Loaded %d ShapeNet-Occ instances", len(self.file_list))
        self.transforms = self._build_transforms(self.split)

    def _parse_line(self, line):
        taxonomy_id = line.split("-")[0].split("/")[-1]
        model_id = line.split("-")[1].split(".")[0]
        if self.cars_only and taxonomy_id != CAR_TAXONOMY_ID:
            return None
        return {
            "taxonomy_id": taxonomy_id,
            "model_id": model_id,
            "file_path": line,
        }

    def _build_transforms(self, split):
        transform_defs = [
            {
                "callback": "RandomSamplePoints",
                "parameters": {"n_points": 2048},
                "objects": ["partial"],
            }
        ]
        if split == "train":
            transform_defs.append({"callback": "RandomMirrorPoints", "objects": ["partial", "gt"]})
        transform_defs.append({"callback": "ToTensor", "objects": ["partial", "gt"]})
        return transforms.Compose(transform_defs)

    @staticmethod
    def normalize_pair(partial, target):
        centroid = np.mean(target, axis=0)
        target = target - centroid
        partial = partial - centroid
        scale = np.max(np.sqrt(np.sum(target**2, axis=1)))
        return partial / scale, target / scale

    def __getitem__(self, idx):
        sample = self.file_list[idx]
        rendering_idx = random.randint(0, self.n_renderings - 1) if self.split == "train" else 0
        partial_path = self.partial_points_path % (sample["taxonomy_id"], sample["model_id"], rendering_idx)
        gt_path = self.gt_path % (sample["taxonomy_id"], sample["model_id"])
        partial = IO.get(partial_path).astype(np.float32)
        target = IO.get(gt_path).astype(np.float32)

        if target.shape[0] != self.npoints:
            raise ValueError(f"Expected {self.npoints} target points in {gt_path}, got {target.shape[0]}")

        partial, target = self.normalize_pair(partial, target)
        data_dict = self.transforms({"partial": partial, "gt": target})
        return {
            "taxonomy_id": sample["taxonomy_id"],
            "model_id": sample["model_id"],
            "source_points": data_dict["partial"],
            "target_points": data_dict["gt"],
        }

    def __len__(self):
        return len(self.file_list)
