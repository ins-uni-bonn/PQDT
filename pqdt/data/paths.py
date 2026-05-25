from pathlib import Path


SHAPENET55_34_DIR = "ShapeNet55-34"


def _get(cfg, key, default=None):
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def shapenet55_34_root(dataset_cfg):
    dataset_dir = _get(dataset_cfg, "dataset_dir")
    if not dataset_dir:
        raise ValueError("dataset.dataset_dir must point to the directory containing ShapeNet55-34/")
    return Path(dataset_dir) / SHAPENET55_34_DIR


def resolve_dataset_path(dataset_cfg, key, default_relative_path):
    explicit_path = _get(dataset_cfg, key)
    if explicit_path:
        return Path(explicit_path)
    return shapenet55_34_root(dataset_cfg) / default_relative_path


def shapenet_split_dir(dataset_cfg):
    return resolve_dataset_path(dataset_cfg, "data_path", "ShapeNet-55")


def shapenet_point_dir(dataset_cfg):
    return resolve_dataset_path(dataset_cfg, "pc_path", "shapenet_pc")


def occlusion_split_dir(dataset_cfg):
    return resolve_dataset_path(dataset_cfg, "data_path", "Occ_ShapeNet_Car_Noise")


def occlusion_partial_points_path(dataset_cfg):
    return str(resolve_dataset_path(dataset_cfg, "partial_points_path", "occ_partial_noise/%s/%s/models/%d.npy"))


def occlusion_gt_path(dataset_cfg):
    return str(resolve_dataset_path(dataset_cfg, "gt_path", "occ_partial_noise/%s/%s/models/gt.npy"))
