import torch
import torch.nn as nn
from einops import rearrange
from pqdt.utils.miscs import SkipFeatureGather


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CrossAttention(nn.Module):
    def __init__(self, dim, out_dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.out_dim = out_dim
        head_dim = out_dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.q_map = nn.Linear(dim, out_dim, bias=qkv_bias)
        self.k_map = nn.Linear(dim, out_dim, bias=qkv_bias)
        self.v_map = nn.Linear(dim, out_dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(out_dim, out_dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, q, v):
        B, N, _ = q.shape
        C = self.out_dim
        k = v
        NK = k.size(1)

        q = self.q_map(q).view(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k_map(k).view(B, NK, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v_map(v).view(B, NK, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    

class GEGroupMultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=None):
        super(GEGroupMultiHeadAttention, self).__init__()
        if d_model % num_heads != 0:
            raise ValueError('`d_model` ({}) must be a multiple of `num_heads` ({}).'.format(d_model, num_heads))

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_model_per_head = d_model // num_heads

        self.proj_q = nn.Linear(self.d_model, self.d_model)
        self.proj_k = nn.Linear(self.d_model, self.d_model)
        self.proj_v = nn.Linear(self.d_model, self.d_model)
        self.proj_p = nn.Linear(self.d_model, self.d_model)
        self.proj_vp = nn.Linear(self.d_model, self.d_model)

        self.dropout = nn.Dropout(dropout) if dropout is not None else nn.Identity()
        self.f_gather = SkipFeatureGather(d_model, tri=False)
        


    def forward(self, q_in, k_in, v_in, embed_qk, group_idx, key_weights=None, key_masks=None, attention_factors=None):
        B, N, C = k_in.size()
        _, _, K = group_idx.size()

        q = self.proj_q(q_in)  # (B, N, C)
        k = self.proj_k(k_in)
        v = self.proj_v(v_in)
        p = self.proj_p(embed_qk)
        vp = self.proj_vp(embed_qk)

        # Efficient gather without expanding
        group_idx_flat = group_idx.view(B, -1)
        k_g = torch.gather(k, 1, group_idx_flat.unsqueeze(-1).expand(-1, -1, C)).view(B, N, K, C)
        v_g = torch.gather(v, 1, group_idx_flat.unsqueeze(-1).expand(-1, -1, C)).view(B, N, K, C)

        # Rearranging
        q = rearrange(q, 'B N (h c) -> B N h c', h=self.num_heads)
        k = rearrange(k_g, 'B N K (h c) -> B N h K c', h=self.num_heads)
        v = rearrange(v_g, 'B N K (h c) -> B N h K c', h=self.num_heads)
        p = rearrange(p, 'B N K (h c) -> B N h K c', h=self.num_heads)
        vp = rearrange(vp, 'B N K (h c) -> B N h K c', h=self.num_heads)

        # Attention
        attention_scores_p = torch.einsum('BNhc,BNhkc->BNhk', q, p)
        attention_scores_e = torch.einsum('BNhc,BNhkc->BNhk', q, k)
        attention_scores = (attention_scores_e + attention_scores_p) / (self.d_model_per_head ** 0.5)

        if attention_factors is not None:
            attention_scores = attention_factors.unsqueeze(1) * attention_scores
        if key_weights is not None:
            attention_scores = attention_scores * key_weights.unsqueeze(1).unsqueeze(1)
        if key_masks is not None:
            attention_scores = attention_scores.masked_fill(key_masks.unsqueeze(1).unsqueeze(1), float('-inf'))

        attention_scores = nn.functional.softmax(attention_scores, dim=-1)
        attention_scores = self.dropout(attention_scores)

        x = torch.einsum('BNhk,BNhkc->BNhc', attention_scores, v + vp)
        x = rearrange(x, 'B N h c -> B N (h c)')

        # Feature gather step
        x = self.f_gather(v_in, x, group_idx)
        return x, attention_scores

