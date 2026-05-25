import argparse

from pqdt.training import build_dataloaders, build_lit_module, build_trainer, load_config, set_seed


def main(cfg):
    set_seed(getattr(cfg, "seed", 42))
    train_loader, val_loader = build_dataloaders(cfg)
    module = build_lit_module(cfg)
    trainer = build_trainer(cfg)
    trainer.fit(module, train_loader, val_loader, ckpt_path=getattr(cfg.checkpoint, "resume_path", None))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/restoration/shapenet_deform.yml")
    args = parser.parse_args()
    main(load_config(args.config))
