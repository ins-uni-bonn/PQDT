from pqdt.training import build_dataloaders, build_lit_module, build_trainer, set_seed


def run_training(cfg):
    set_seed(getattr(cfg, "seed", 42))
    train_loader, val_loader = build_dataloaders(cfg)
    module = build_lit_module(cfg)
    trainer = build_trainer(cfg)
    trainer.fit(module, train_loader, val_loader, ckpt_path=getattr(cfg.checkpoint, "resume_path", None))
