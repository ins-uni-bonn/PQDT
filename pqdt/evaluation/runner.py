import json
import logging
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from pqdt.data.deformed_shapenet import deform_point_cloud_with_gabor_batched
from pqdt.data.batch import crop_directions
from pqdt.data.paths import occlusion_gt_path, occlusion_partial_points_path
from pqdt.tasks import RestorationLitModule
from pqdt.training import build_test_dataloader
from pqdt.utils.average_meter import AverageMeter
from pqdt.utils.metrics import Metrics
from pqdt.utils.miscs import fps_subsample, separate_point_cloud


LOGGER = logging.getLogger(__name__)


CROP_RATIO = {
    "easy": 1 / 4,
    "median": 1 / 2,
    "hard": 3 / 4,
}


def run_evaluation(cfg, ckpt_path=None):
    test_loader = build_test_dataloader(cfg)
    checkpoint_path = ckpt_path or cfg.checkpoint.test_path
    if not checkpoint_path:
        raise ValueError("No checkpoint path provided. Set checkpoint.test_path or pass --ckpt.")

    task = RestorationLitModule.load_from_checkpoint(checkpoint_path, map_location="cpu")
    model = task.model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    evaluate_model(model, test_loader, cfg, device)


def pc_norm(partial, gt):
    centroid = np.mean(gt, axis=0)
    gt = gt - centroid
    partial = partial - centroid
    scale = np.max(np.sqrt(np.sum(gt**2, axis=1)))
    gt = gt / scale
    partial = partial / scale
    return partial, gt


def evaluate_model(model, test_dataloader, cfg, device):
    test_metrics = AverageMeter(Metrics.names())
    category_metrics = {}
    n_samples = len(test_dataloader)

    with torch.no_grad():
        for batch in tqdm(test_dataloader, total=n_samples):
            taxonomy_ids = batch["taxonomy_id"]
            model_ids = batch["model_id"]
            targets = batch["target_points"]
            dataset_name = cfg.dataset.name

            for sample_idx, taxonomy_id in enumerate(taxonomy_ids):
                taxonomy_id = taxonomy_id if isinstance(taxonomy_id, str) else taxonomy_id.item()
                model_id = model_ids[sample_idx]
                if dataset_name == "shapenet_occ":
                    _evaluate_occ_sample(model, cfg, taxonomy_id, model_id, category_metrics, device)
                elif dataset_name == "shapenet_deform":
                    gt = targets[sample_idx : sample_idx + 1].to(device)
                    _evaluate_cropped_sample(model, cfg, taxonomy_id, gt, category_metrics, device, deform=True)
                elif dataset_name == "shapenet55":
                    gt = targets[sample_idx : sample_idx + 1].to(device)
                    _evaluate_cropped_sample(model, cfg, taxonomy_id, gt, category_metrics, device, deform=False)
                else:
                    raise NotImplementedError(f"Test phase does not support {dataset_name}")

        for meter in category_metrics.values():
            test_metrics.update(meter.avg())

    print("[TEST] Metrics = %s" % (["%.4f" % metric for metric in test_metrics.avg()]))
    _print_results(test_metrics, category_metrics)


def _evaluate_occ_sample(model, cfg, taxonomy_id, model_id, category_metrics, device):
    idx_bias = {"easy": 0, "median": 32, "hard": 64}
    choice = [view_idx + idx_bias[cfg.dataset.crop_mode] for view_idx in range(32)]
    partial_points_path = occlusion_partial_points_path(cfg.dataset)
    gt_points_path = occlusion_gt_path(cfg.dataset)
    for view_idx in choice:
        partial_path = partial_points_path % (taxonomy_id, model_id, view_idx)
        partial = np.load(partial_path).astype(np.float32)
        gt_path = gt_points_path % (taxonomy_id, model_id)
        gt = np.load(gt_path).astype(np.float32)
        partial, gt = pc_norm(partial, gt)
        if partial.shape[0] != 2048:
            raise ValueError(f"Expected 2048 partial points, got {partial.shape[0]} from {partial_path}")

        partial = torch.from_numpy(partial).unsqueeze(0).to(device)
        gt = torch.from_numpy(gt).unsqueeze(0).to(device)

        _, outputs = model(partial)
        _, _, _, u3 = outputs
        metrics = Metrics.get(u3, gt, require_emd=False)
        _update_category_metrics(category_metrics, taxonomy_id, metrics)


def _evaluate_cropped_sample(model, cfg, taxonomy_id, gt, category_metrics, device, deform):
    npoints = cfg.dataset.n_points
    num_crop = int(npoints * CROP_RATIO[cfg.dataset.crop_mode])
    for direction in crop_directions(device):
        source = gt
        if deform:
            source = deform_point_cloud_with_gabor_batched(
                gt,
                scale=0.4,
                num_kernels=16,
                freq=2.0,
                sigma=0.5,
                seed=42,
            )

        partial, _ = separate_point_cloud(source, npoints, num_crop, fixed_points=direction)
        partial = fps_subsample(partial, 2048)
        _, outputs = model(partial)
        _, _, _, u3 = outputs
        pred = torch.cat([partial, u3], dim=1) if not deform else u3
        metrics = Metrics.get(pred, gt, require_emd=deform)
        _update_category_metrics(category_metrics, taxonomy_id, metrics)


def _update_category_metrics(category_metrics, taxonomy_id, metrics):
    if taxonomy_id not in category_metrics:
        category_metrics[taxonomy_id] = AverageMeter(Metrics.names())
    category_metrics[taxonomy_id].update(metrics)


def _print_results(test_metrics, category_metrics):
    synset_names = _load_synset_names()
    print("============================ TEST RESULTS ============================")
    msg = "Taxonomy\t#Sample\t"
    for metric in test_metrics.items:
        msg += metric + "\t"
    msg += "#ModelName\t"
    print(msg)

    for taxonomy_id in category_metrics:
        msg = taxonomy_id + "\t"
        msg += str(category_metrics[taxonomy_id].count(0)) + "\t"
        for value in category_metrics[taxonomy_id].avg():
            msg += "%.3f \t" % value
        msg += synset_names.get(taxonomy_id, taxonomy_id) + "\t"
        print(msg)

    msg = "Overall \t\t"
    for value in test_metrics.avg():
        msg += "%.3f \t" % value
    print(msg)


def _load_synset_names():
    candidates = [
        Path("pqdt/data/shapenet_synset_dict.json"),
        Path("data/shapenet_synset_dict.json"),
    ]
    for path in candidates:
        if path.exists():
            with path.open("r") as f:
                return json.load(f)
    return {}
