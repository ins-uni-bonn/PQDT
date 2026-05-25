import math

import torch


def _get(cfg, key, default=None):
    return getattr(cfg, key, default)


def build_warm_cosine_scheduler(optimizer, lr_max, lr_min, warmup_epochs, max_epochs):
    def lr_lambda(epoch):
        if warmup_epochs > 0 and epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs

        denom = max(1, max_epochs - warmup_epochs)
        cos_decay = 0.5 * (1 + math.cos(math.pi * (epoch - warmup_epochs) / denom))
        return (lr_min / lr_max) + (1 - lr_min / lr_max) * cos_decay

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def configure_optimizers_from_config(module, cfg):
    optimizer_cfg = cfg.optimizer
    name = _get(optimizer_cfg, "name", "adam").lower()
    lr = optimizer_cfg.lr
    weight_decay = _get(optimizer_cfg, "weight_decay", 0.0)
    fused = bool(_get(optimizer_cfg, "fused", False))
    params = module.parameters()

    if name == "adamw":
        optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay, fused=fused)
    elif name == "adam":
        optimizer = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay, fused=fused)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_cfg.name}")

    scheduler_cfg = getattr(cfg, "scheduler", None)
    if scheduler_cfg is None or _get(scheduler_cfg, "name", "none").lower() in {"none", "null"}:
        return optimizer

    scheduler_name = _get(scheduler_cfg, "name", "warm_cosine").lower()
    if scheduler_name != "warm_cosine":
        raise ValueError(f"Unsupported scheduler: {scheduler_cfg.name}")

    scheduler = build_warm_cosine_scheduler(
        optimizer,
        lr_max=lr,
        lr_min=scheduler_cfg.end_lr,
        warmup_epochs=scheduler_cfg.warmup_epochs,
        max_epochs=cfg.trainer.max_epochs,
    )
    return [optimizer], [scheduler]
