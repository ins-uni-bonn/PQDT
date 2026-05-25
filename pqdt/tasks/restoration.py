from pqdt.utils.torchvision_compat import ensure_torchvision_nms_schema

ensure_torchvision_nms_schema()

import lightning as L
from easydict import EasyDict as edict

from extensions.chamfer_dist import ChamferDistanceL1
from pqdt.training import to_plain_dict
from pqdt.data.batch import prepare_restoration_batch
from pqdt.models import build_model
from pqdt.training.losses import compute_pq_reconstruction_losses
from pqdt.training.optim import configure_optimizers_from_config
from pqdt.training.pointcloud_logging import log_reconstruction_meshes
from pqdt.utils.metrics import Metrics


class RestorationLitModule(L.LightningModule):
    """Lightning task for training PQDT restoration models."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = edict(cfg) if isinstance(cfg, dict) else cfg
        self.save_hyperparameters({"cfg": to_plain_dict(self.cfg)})
        self.model = build_model(self.cfg)
        self.loss_func = ChamferDistanceL1(hyper=False)
        self.log_pcd_cnt = 1

    def forward(self, x):
        self.model.set_epoch(int(self.current_epoch))
        return self.model(x)

    def training_step(self, batch, batch_idx):
        batch = prepare_restoration_batch(batch, self.cfg, "train")
        src_pcd = batch["source_points"]
        tgt_pcd = batch["target_points"]
        self.model.set_epoch(int(self.current_epoch))
        pq, outputs = self.model(src_pcd)
        losses = compute_pq_reconstruction_losses(self.loss_func, pq, outputs, tgt_pcd)

        self.log_pcd_cnt = 0
        self.log("train/loss", losses["loss"], prog_bar=True)
        self.log("train/loss_pq", losses["loss_pq"])
        self.log("train/loss_fine", losses["loss_fine"], prog_bar=True)

        if self.global_step % 500 == 0:
            self._log_visualizations("train", src_pcd, tgt_pcd, outputs[-1], pq)

        return losses["loss"]

    def validation_step(self, batch, batch_idx):
        batch = prepare_restoration_batch(batch, self.cfg, "val")
        src_pcd = batch["source_points"]
        tgt_pcd = batch["target_points"]
        self.model.set_epoch(int(self.current_epoch))
        pq, outputs = self.model(src_pcd)
        fine = outputs[-1]

        metrics = Metrics.get(fine, tgt_pcd, require_emd=False)
        f_score, cd1, cd2, _ = [metric.item() for metric in metrics]

        self.log("val/f_score", f_score, prog_bar=True)
        self.log("val/cd1", cd1, prog_bar=True)
        self.log("val/cd2", cd2, prog_bar=True)

        if self.log_pcd_cnt == 0:
            self.log_pcd_cnt += 1
            self._log_visualizations("val", src_pcd, tgt_pcd, fine, pq)

    def test_step(self, batch, batch_idx):
        batch = prepare_restoration_batch(batch, self.cfg, "test")
        src_pcd = batch["source_points"]
        tgt_pcd = batch["target_points"]
        self.model.set_epoch(int(self.current_epoch))
        _, outputs = self.model(src_pcd)
        metrics = Metrics.get(outputs[-1], tgt_pcd, require_emd=False)
        f_score, cd1, cd2, _ = [metric.item() for metric in metrics]
        self.log("test/f_score", f_score, prog_bar=True)
        self.log("test/cd1", cd1, prog_bar=True)
        self.log("test/cd2", cd2, prog_bar=True)

    def configure_optimizers(self):
        return configure_optimizers_from_config(self, self.cfg)

    def _log_visualizations(self, mode, src_pcd, tgt_pcd, rebuild_points, coarse_pcd):
        log_reconstruction_meshes(
            self.logger,
            self.global_step,
            mode,
            tgt_pcd,
            src_pcd,
            rebuild_points,
            coarse_pcd,
            up_factors=self.cfg.model.up_factors,
        )
