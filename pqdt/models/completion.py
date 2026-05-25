from torch import nn

from pqdt.models.pq_transformer import PQTransformer
from pqdt.models.stem import PQStemEncoder
from pqdt.models.uptrans import UpLayer


def _get(cfg, key, default=None):
    return getattr(cfg, key, default)


def build_pq_transformer(model_cfg, total_epochs):
    if model_cfg.name.lower() != "pqdt":
        raise ValueError(f"Unsupported PQ transformer: {model_cfg.name}")

    return PQTransformer(
        embed_dim=model_cfg.trans_dim,
        num_heads=model_cfg.num_heads,
        enc_attn=model_cfg.enc_attn,
        dec_attn=model_cfg.dec_attn,
        num_pseudo=model_cfg.num_pseudo,
        num_query=model_cfg.num_queries,
        tau0=model_cfg.tau0,
        total_epochs=total_epochs,
        r_sph=model_cfg.r_sph,
        in_q=model_cfg.in_q,
    )


class PQCompletionModel(nn.Module):
    """Pure point-cloud completion network without training-loop responsibilities."""

    def __init__(self, model_cfg, total_epochs):
        super().__init__()
        stem_cfg = _get(model_cfg, "stem_encoder")
        self.model_cfg = model_cfg
        self.trans_dim = model_cfg.trans_dim
        self.up_factors = model_cfg.up_factors
        self.stem_encoder = PQStemEncoder(
            in_chans=_get(stem_cfg, "in_chans", 256),
            embed_dim=model_cfg.trans_dim,
            num_heads=model_cfg.num_heads,
            enc_attn=_get(stem_cfg, "enc_attn", model_cfg.enc_attn),
            transdown_fps=_get(stem_cfg, "transdown_fps", (512, 128)),
            transdown_dims=_get(stem_cfg, "transdown_dims", (64, 256)),
            transdown_num_heads=_get(stem_cfg, "transdown_num_heads", (1, 4)),
            transdown_sa_depth=_get(stem_cfg, "transdown_sa_depth", (3, 3)),
            transdown_k=_get(stem_cfg, "transdown_k", (16, 16)),
        )
        self.transformer = build_pq_transformer(model_cfg, total_epochs)
        self.up_layers = nn.ModuleList(
            [
                UpLayer(
                    dim=self.trans_dim,
                    seed_dim=self.trans_dim,
                    up_factor=factor,
                    i=index,
                    n_knn=model_cfg.up_n_knn,
                    radius=model_cfg.up_radius,
                    interpolate=model_cfg.up_interpolate,
                    attn_channel=model_cfg.up_attn_channel,
                )
                for index, factor in enumerate(self.up_factors)
            ]
        )

    def set_epoch(self, epoch):
        self.transformer.set_epoch(epoch)

    def down(self, points):
        return self.stem_encoder.down(points)

    def backbone(self, stem_output):
        return self.transformer(stem_output)

    def up(self, coor_pq, coor_c, f_c):
        seed = coor_c
        predictions = [seed.transpose(1, 2).contiguous()]
        points = seed
        previous_knn = None
        for layer in self.up_layers:
            points, previous_knn = layer(points, seed, f_c, previous_knn)
            predictions.append(points.transpose(1, 2).contiguous())
        return coor_pq.transpose(2, 1).contiguous(), predictions

    def forward(self, points):
        stem_output = self.stem_encoder(points)
        seed, coor_c, f_c = self.backbone(stem_output)
        return self.up(seed, coor_c, f_c)

    def forward_dual(self, source_points, sketch_points):
        stem_output = self.stem_encoder.forward_dual(source_points, sketch_points)
        seed, coor_c, f_c = self.backbone(stem_output)
        return self.up(seed, coor_c, f_c)


def build_pq_completion_model(cfg):
    return PQCompletionModel(cfg.model, total_epochs=cfg.trainer.max_epochs)
