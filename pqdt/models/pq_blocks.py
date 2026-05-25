import torch
import torch.nn as nn

from pqdt.utils.attention import Attention, CrossAttention, GEGroupMultiHeadAttention
from pqdt.utils.layers import DropPath
from pqdt.utils.miscs import GumbelTopK, Mlp, get_knn_index, get_tau
from pqdt.utils.positional_encoding import GeometricStructureEmbedding


class GEEncoder(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        attn_cls,
        mlp_ratio=4.0,
        drop=0.0,
        attn_drop=None,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        pos_add=True,
    ):
        super().__init__()
        self.pos_embed = nn.Sequential(
            nn.Conv1d(3, 128, 1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(128, dim, 1),
        )
        self.pos_embed_geo = GeometricStructureEmbedding(dim, sigma_d=0.2, sigma_a=15, angle_k=3)
        self.encoder = nn.ModuleList(
            [
                GEEncoderBlock(
                    dim=dim,
                    num_heads=num_heads,
                    attn=attn,
                    mlp_ratio=mlp_ratio,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path,
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                )
                for attn in attn_cls
            ]
        )
        self.pos_add = pos_add

    def forward(self, coor, features):
        _, group_idx = get_knn_index(coor, k=12)
        geo_pos = self.pos_embed_geo(coor.transpose(1, 2).contiguous(), group_idx)
        pos = self.pos_embed(coor).transpose(1, 2)
        if self.pos_add:
            features = features + pos
        for block in self.encoder:
            features = block(features, group_idx, geo_pos)
        return features


class GEEncoderBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        attn="ge_attn",
        mlp_ratio=4.0,
        drop=0.0,
        attn_drop=None,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn_class = attn
        if attn == "ge_attn":
            self.attn = GEGroupMultiHeadAttention(dim, num_heads=num_heads, dropout=attn_drop)
        elif attn == "attn":
            self.attn = Attention(dim, num_heads=num_heads)
        else:
            raise ValueError(f"Unsupported attention type: {attn}")

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)

    def forward(self, x, group_idx, geo_pos):
        norm_x = self.norm1(x)
        if self.attn_class == "ge_attn":
            update, _ = self.attn(norm_x, norm_x, norm_x, geo_pos, group_idx)
        else:
            update = self.attn(norm_x)

        x = x + self.drop_path(update)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class GEDecoder(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        attn_cls,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=None,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        pos_add=True,
    ):
        super().__init__()
        self.pos_embed = nn.Sequential(
            nn.Conv1d(3, 128, 1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(128, dim, 1),
        )
        self.pos_embed_geo = GeometricStructureEmbedding(dim, sigma_d=0.2, sigma_a=15, angle_k=3)
        self.decoder = nn.ModuleList(
            [
                GEDecoderBlock(
                    dim=dim,
                    num_heads=num_heads,
                    attn=attn,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path,
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                )
                for attn in attn_cls
            ]
        )
        self.pos_add = pos_add

    def forward(self, coor_q, query_features, coor_x, memory_features):
        del coor_x
        _, group_idx_q = get_knn_index(coor_q, k=12)
        geo_pos_q = self.pos_embed_geo(coor_q.transpose(1, 2).contiguous(), group_idx_q)
        pos_q = self.pos_embed(coor_q).transpose(1, 2)
        if self.pos_add:
            query_features = query_features + pos_q
        for block in self.decoder:
            query_features = block(query_features, memory_features, group_idx_q, geo_pos_q)
        return query_features


class GEDecoderBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        attn="ge_attn",
        dim_q=None,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn_class = attn
        if attn == "ge_attn":
            self.self_attn = GEGroupMultiHeadAttention(dim, num_heads=num_heads, dropout=attn_drop)
        elif attn == "attn":
            self.self_attn = Attention(dim, num_heads=num_heads)
        else:
            raise ValueError(f"Unsupported attention type: {attn}")

        dim_q = dim_q or dim
        self.norm_q = norm_layer(dim_q)
        self.norm_v = norm_layer(dim)
        self.cross_attn = CrossAttention(
            dim,
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)

    def forward(self, query, memory, group_idx_q, geo_pos_q):
        norm_query = self.norm1(query)
        if self.attn_class == "ge_attn":
            update, _ = self.self_attn(norm_query, norm_query, norm_query, geo_pos_q, group_idx_q)
        else:
            update = self.self_attn(norm_query)

        query = query + self.drop_path(update)
        query = query + self.drop_path(self.cross_attn(self.norm_q(query), self.norm_v(memory)))
        query = query + self.drop_path(self.mlp(self.norm2(query)))
        return query


class DQS(nn.Module):
    def __init__(self, embed_dim=384, gf_dim=1024, num_query=384, tau0=1.0, total_epochs=200, sel=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_query = num_query
        self.tau0 = tau0
        self.total_epochs = total_epochs
        self.current_epoch = 0
        self.mlp_query = nn.Sequential(
            nn.Conv1d(3 + embed_dim + gf_dim, 1024, 1),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(1024, 1024, 1),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(1024, embed_dim, 1),
        )
        self.sel = sel
        self.sampler = GumbelTopK(k=num_query)
        self.query_ranking = nn.Sequential(
            nn.Conv1d(embed_dim, 512, kernel_size=1),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Conv1d(512, 128, kernel_size=1),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Conv1d(128, 1, kernel_size=1),
        )

    def forward(self, coor, features, global_features):
        batch_size, num_points, _ = features.size()
        aggregated = torch.cat(
            [
                features.transpose(1, 2),
                global_features.unsqueeze(-1).expand(-1, -1, num_points),
                coor,
            ],
            dim=1,
        )
        aggregated = self.mlp_query(aggregated).transpose(1, 2)
        if not self.sel:
            return coor, aggregated, features

        scores = self.query_ranking(aggregated.transpose(1, 2)).squeeze(1)
        scores = (scores - scores.mean(-1, keepdim=True)) / (scores.std(-1, keepdim=True) + 1e-8)
        if self.num_query > scores.shape[1]:
            raise ValueError(f"num_query={self.num_query} exceeds available points={scores.shape[1]}")

        self.sampler.noise_scale = get_tau(
            self.current_epoch,
            tau0=self.tau0,
            tau_min=0.0,
            total_epochs=self.total_epochs,
            mode="cosine",
        )
        indices = self.sampler(scores)
        coor = torch.gather(coor, 2, indices.unsqueeze(1).expand(batch_size, 3, -1))
        aggregated = torch.gather(
            aggregated,
            1,
            indices.unsqueeze(-1).expand(-1, -1, aggregated.size(-1)),
        )
        features = torch.gather(features, 1, indices.unsqueeze(-1).expand(-1, -1, features.size(-1)))
        return coor, aggregated, features


class SoftBaseHead(nn.Module):
    def __init__(self, d_model: int, radius: float = 0.3):
        super().__init__()
        self.radius = radius
        self.proj_q = nn.Linear(d_model, d_model)
        self.proj_k = nn.Linear(d_model, d_model)
        self.mlp_res = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, 3),
        )

    def forward(self, decoder_output, seed_features, seed_xyz, seed_mask=None):
        query = self.proj_q(decoder_output.float())
        key = self.proj_k(seed_features.float())
        attn = torch.matmul(query, key.transpose(1, 2)) / (query.size(-1) ** 0.5)
        if seed_mask is not None:
            attn = attn.masked_fill(~seed_mask.unsqueeze(1), float("-inf"))
        weights = attn.softmax(dim=-1)

        seed_xyz = seed_xyz.transpose(1, 2).contiguous()
        base = torch.bmm(weights, seed_xyz.float())
        delta = torch.tanh(self.mlp_res(decoder_output.float())) * self.radius
        xyz = base + delta
        return xyz.transpose(1, 2).contiguous(), weights
