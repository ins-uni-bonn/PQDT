from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.utils.data as torch_data
import yaml
from easydict import EasyDict as edict


def _get(cfg, key, default=None):
    return getattr(cfg, key, default)


def load_config(path: str | Path) -> edict:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file does not exist: {config_path}")

    with config_path.open("r") as f:
        return edict(yaml.safe_load(f))


def to_plain_dict(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: to_plain_dict(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_plain_dict(item) for item in value]
    return value


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def build_dataset(cfg, split):
    from pqdt.data import build_dataset as _build_dataset

    return _build_dataset(cfg, split)


def _build_loader(dataset, batch_size, shuffle, num_workers, pin_memory, persistent_workers):
    return torch_data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers if num_workers > 0 else False,
    )


def build_dataloaders(cfg):
    dataloader_cfg = cfg.dataloader
    train_split = _get(cfg.dataset, "train_split", "train")
    val_split = _get(cfg.dataset, "val_split", "test")
    num_workers = int(_get(dataloader_cfg, "num_workers", 0))
    val_num_workers = int(_get(dataloader_cfg, "val_num_workers", num_workers))
    pin_memory = bool(_get(dataloader_cfg, "pin_memory", True))
    persistent_workers = bool(_get(dataloader_cfg, "persistent_workers", num_workers > 0))

    train_loader = _build_loader(
        build_dataset(cfg, train_split),
        batch_size=dataloader_cfg.train_batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
    val_loader = _build_loader(
        build_dataset(cfg, val_split),
        batch_size=dataloader_cfg.val_batch_size,
        shuffle=False,
        num_workers=val_num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
    return train_loader, val_loader


def build_test_dataloader(cfg):
    dataloader_cfg = cfg.dataloader
    test_split = _get(cfg.dataset, "test_split", "test")
    num_workers = int(_get(dataloader_cfg, "test_num_workers", _get(dataloader_cfg, "val_num_workers", 0)))
    pin_memory = bool(_get(dataloader_cfg, "pin_memory", True))
    persistent_workers = bool(_get(dataloader_cfg, "persistent_workers", num_workers > 0))
    return _build_loader(
        build_dataset(cfg, test_split),
        batch_size=dataloader_cfg.test_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )


def _task_name(cfg):
    task = cfg.task
    return task if isinstance(task, str) else getattr(task, "name")


def build_lit_module(cfg):
    task = _task_name(cfg).lower()
    model_name = cfg.model.name.lower()
    if task == "restoration" and model_name == "pqdt":
        from pqdt.tasks import RestorationLitModule

        return RestorationLitModule(cfg)
    raise ValueError(f"Unsupported task/model pair: {_task_name(cfg)}/{cfg.model.name}")


def _checkpoint_callback(cfg):
    from lightning.pytorch.callbacks import ModelCheckpoint

    checkpoint_cfg = cfg.checkpoint
    return ModelCheckpoint(
        filename=f"{cfg.model.name}-{cfg.dataset.name}-{{epoch:02d}}",
        dirpath=checkpoint_cfg.dirpath,
        monitor=checkpoint_cfg.monitor,
        mode=checkpoint_cfg.mode,
        save_top_k=checkpoint_cfg.save_top_k,
        save_last=bool(_get(checkpoint_cfg, "save_last", True)),
        every_n_epochs=_get(checkpoint_cfg, "every_n_epochs", cfg.trainer.check_val_every_n_epoch),
    )


def build_trainer(cfg):
    from pqdt.utils.torchvision_compat import ensure_torchvision_nms_schema

    ensure_torchvision_nms_schema()

    import lightning as L
    from lightning.pytorch import loggers as pl_loggers
    from lightning.pytorch.callbacks import LearningRateMonitor, ModelSummary

    trainer_cfg = cfg.trainer
    logger_cfg = cfg.logger
    tb_logger = pl_loggers.TensorBoardLogger(
        save_dir=logger_cfg.log_dir,
        name=_get(logger_cfg, "name", f"{_task_name(cfg)}-{cfg.model.name}-{cfg.dataset.name}"),
    )
    callbacks = [
        LearningRateMonitor(logging_interval="epoch"),
        _checkpoint_callback(cfg),
        ModelSummary(max_depth=int(_get(trainer_cfg, "model_summary_depth", 2))),
    ]
    trainer_kwargs = {
        "logger": tb_logger,
        "log_every_n_steps": trainer_cfg.log_every_n_steps,
        "check_val_every_n_epoch": trainer_cfg.check_val_every_n_epoch,
        "callbacks": callbacks,
        "max_epochs": trainer_cfg.max_epochs,
        "accelerator": _get(trainer_cfg, "accelerator", "auto"),
        "devices": _get(trainer_cfg, "devices", "auto"),
    }
    for optional_key in ("limit_train_batches", "limit_val_batches", "precision"):
        if hasattr(trainer_cfg, optional_key):
            trainer_kwargs[optional_key] = getattr(trainer_cfg, optional_key)
    return L.Trainer(**trainer_kwargs)
