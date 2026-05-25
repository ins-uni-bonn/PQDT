from importlib import import_module


DATASETS = {
    "shapenet55": "pqdt.data.shapenet55:ShapeNet55Dataset",
    "shapenet_deform": "pqdt.data.deformed_shapenet:DeformedShapeNetDataset",
    "shapenet_occ": "pqdt.data.occluded_shapenet:OccludedShapeNetDataset",
}


def _load_class(spec):
    module_name, class_name = spec.split(":")
    return getattr(import_module(module_name), class_name)


def build_dataset(cfg, split):
    try:
        dataset_cls = _load_class(DATASETS[cfg.dataset.name])
    except KeyError as exc:
        valid = ", ".join(sorted(DATASETS))
        raise ValueError(f"Unknown dataset '{cfg.dataset.name}'. Available datasets: {valid}") from exc
    return dataset_cls(cfg, split)


__all__ = ["DATASETS", "build_dataset"]
