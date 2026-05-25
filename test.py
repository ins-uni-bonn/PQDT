import argparse

from pqdt.training import load_config
from pqdt.evaluation import run_evaluation


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/restoration/shapenet_deform.yml")
    parser.add_argument("--ckpt", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_evaluation(cfg, ckpt_path=args.ckpt)
