import torch
import torch.nn as nn

from pqdt.models.pq_blocks import DQS, GEDecoder, GEEncoder, SoftBaseHead
from pqdt.utils.miscs import fps_subsample, sample_sphere


class PQTransformer(nn.Module):
    """Pseudo-query dual transformer that consumes a precomputed PQ stem output."""

    def __init__(
        self,
        embed_dim=384,
        num_heads=6,
        enc_attn=("ge_attn", "attn", "attn", "attn"),
        dec_attn=("ge_attn", "attn", "attn", "attn", "attn", "attn", "attn", "attn"),
        mlp_ratio=2.0,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        num_pseudo=384,
        num_query=512,
        tau0=1.0,
        total_epochs=200,
        r_sph=0.8,
        in_q=False,
    ):
        super().__init__()
        self.in_q = in_q
        self.r_sph = r_sph
        self.encoder_2 = GEEncoder(
            embed_dim,
            num_heads,
            attn_cls=enc_attn,
            mlp_ratio=mlp_ratio,
            drop=drop_rate,
            attn_drop=attn_drop_rate,
        )
        self.decoder_1 = GEDecoder(
            embed_dim,
            num_heads,
            attn_cls=dec_attn[:4],
            mlp_ratio=mlp_ratio,
            drop=drop_rate,
            attn_drop=attn_drop_rate,
        )
        self.decoder_2 = GEDecoder(
            embed_dim,
            num_heads,
            attn_cls=dec_attn,
            mlp_ratio=mlp_ratio,
            drop=drop_rate,
            attn_drop=attn_drop_rate,
        )
        self.increase_dim_1 = nn.Sequential(
            nn.Linear(embed_dim, 1024),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(1024, 1024),
        )
        self.increase_dim_2 = nn.Sequential(
            nn.Linear(embed_dim, 1024),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(1024, 1024),
        )
        self.increase_dim_3 = nn.Sequential(
            nn.Linear(embed_dim, 1024),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(1024, 1024),
        )
        self.num_query = num_query
        self.num_pseudo = num_pseudo
        self.qf_1 = DQS(embed_dim=embed_dim, gf_dim=1024, num_query=num_pseudo, tau0=tau0, total_epochs=total_epochs)
        self.qf_2 = DQS(embed_dim=embed_dim, gf_dim=1024, num_query=num_query, tau0=tau0, total_epochs=total_epochs)
        self.mlp_query = nn.Sequential(
            nn.Conv1d(1024 + 1024 + 3, 1024, 1),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(1024, 1024, 1),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(1024, embed_dim, 1),
        )
        self.mlp_query_ps = nn.Sequential(
            nn.Conv1d(1024 + 3, 1024, 1),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(1024, 1024, 1),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(1024, embed_dim, 1),
        )
        self.softbase = SoftBaseHead(d_model=embed_dim, radius=1.0)
        self.reduce_map = nn.Conv1d(embed_dim + 1027, embed_dim, 1)

    def set_epoch(self, epoch):
        self.qf_1.current_epoch = epoch
        self.qf_2.current_epoch = epoch

    def forward(self, stem_output):
        coor_in = stem_output.coors[0]
        coor_c = stem_output.coor_c
        x1 = stem_output.x1

        x1_g = self.increase_dim_1(x1)
        x1_g = torch.max(x1_g, dim=1)[0]

        if self.in_q:
            coor_ps = fps_subsample(coor_in.transpose(1, 2).contiguous(), self.num_pseudo)
            coor_ps = coor_ps.transpose(1, 2).contiguous()
        else:
            coor_ps = fps_subsample(sample_sphere(2048, x1.shape[0], self.r_sph, x1.device), self.num_pseudo)
            coor_ps = coor_ps.transpose(1, 2).contiguous()

        q_ps = torch.cat([x1_g.unsqueeze(-1).expand(-1, -1, self.num_pseudo), coor_ps], dim=1)
        q_ps = self.mlp_query_ps(q_ps).transpose(1, 2)
        q_ps = self.decoder_1(coor_ps, q_ps, coor_c, x1)
        pseudo_seed, _ = self.softbase(q_ps, q_ps, coor_ps, seed_mask=None)

        x1_ps = torch.cat([x1, q_ps], dim=1)
        pseudo_seed = torch.cat([coor_c, pseudo_seed], dim=-1)
        pseudo_seed_sel, x1_ps, x_res = self.qf_1(pseudo_seed, x1_ps, x1_g)

        x2 = self.encoder_2(pseudo_seed_sel, x1_ps)
        x2_g = self.increase_dim_2(x2)
        x2_g = torch.max(x2_g, dim=1)[0]
        x2 = x2 + x_res

        seed_fps = fps_subsample(coor_in.transpose(1, 2).contiguous(), self.num_query // 2)
        seed_fps = seed_fps.transpose(1, 2).contiguous()
        zeros = torch.zeros(x2.shape[0], seed_fps.shape[-1], x2.shape[-1], device=x2.device)
        x2_q = torch.cat([x2, zeros], dim=1)
        query_seed = torch.cat([pseudo_seed_sel, seed_fps], dim=-1)
        query_seed, x2_q, _ = self.qf_2(query_seed, x2_q, x2_g)

        query = torch.cat(
            [
                x1_g.unsqueeze(-1).expand(-1, -1, self.num_query),
                x2_g.unsqueeze(-1).expand(-1, -1, self.num_query),
                query_seed,
            ],
            dim=1,
        )
        query = self.mlp_query(query).transpose(1, 2)
        query = self.decoder_2(query_seed, query, pseudo_seed_sel, x2_q)
        query_g = self.increase_dim_3(query)
        query_g = torch.max(query_g, dim=1)[0]

        query_features = torch.cat(
            [
                query_g.unsqueeze(-1).expand(-1, -1, self.num_query),
                query.transpose(1, 2),
                query_seed,
            ],
            dim=1,
        )
        query_features = self.reduce_map(query_features)
        return pseudo_seed, query_seed, query_features
