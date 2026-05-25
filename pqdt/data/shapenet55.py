import logging

import numpy as np
import torch
import torch.utils.data as data
from easydict import EasyDict as edict

from pqdt.data.io import IO
from pqdt.data.paths import shapenet_point_dir, shapenet_split_dir


LOGGER = logging.getLogger(__name__)
CAR_TAXONOMY_ID = "02958343"


def _as_cfg(cfg):
    return edict(cfg) if isinstance(cfg, dict) else cfg


def normalize_point_cloud(points):
    centroid = np.mean(points, axis=0)
    points = points - centroid
    scale = np.max(np.sqrt(np.sum(points**2, axis=1)))
    return points / scale


class BaseShapeNetPointDataset(data.Dataset):
    def __init__(self, cfg, split, cars_only=False):
        super().__init__()
        cfg = _as_cfg(cfg)
        dataset_cfg = cfg.dataset
        self.data_root = shapenet_split_dir(dataset_cfg)
        self.pc_path = shapenet_point_dir(dataset_cfg)
        self.npoints = int(dataset_cfg.n_points)
        self.split = split
        self.cars_only = bool(cars_only)
        self.data_list_file = self.data_root / f"{self.split}.txt"

        LOGGER.info("Opening ShapeNet split file: %s", self.data_list_file)
        with self.data_list_file.open("r") as f:
            lines = [line.strip() for line in f if line.strip()]

        self.file_list = [sample for sample in (self._parse_line(line) for line in lines) if sample is not None]
        LOGGER.info("Loaded %d ShapeNet instances", len(self.file_list))

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

    def __getitem__(self, idx):
        sample = self.file_list[idx]
        points = IO.get(str(self.pc_path / sample["file_path"])).astype(np.float32)
        points = normalize_point_cloud(points)
        return {
            "taxonomy_id": sample["taxonomy_id"],
            "model_id": sample["model_id"],
            "target_points": torch.from_numpy(points).float(),
        }

    def __len__(self):
        return len(self.file_list)


class ShapeNet55Dataset(BaseShapeNetPointDataset):
    """ShapeNet55/34 target point clouds used for crop-completion restoration."""

    def __init__(self, cfg, split):
        super().__init__(cfg, split, cars_only=False)
