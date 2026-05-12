# Adapted from https://github.com/lucidrains/enformer-pytorch/tree/main
#
# MIT License
#
# Copyright (c) 2021 Phil Wang, 2024 Johannes Hingerl

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# =========================================================================

import math

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import einsum, nn


def get_positional_features_central_mask(positions, features, seq_len):
    pow_rate = math.exp(math.log(seq_len + 1) / features)
    center_widths = torch.pow(
        pow_rate, torch.arange(1, features + 1, device=positions.device)
    ).float()
    center_widths = center_widths - 1
    return (center_widths[None, ...] > positions.abs()[..., None]).float()


def get_positional_embed(seq_len, feature_size, device):
    distances = torch.arange(-seq_len + 1, seq_len, device=device)

    feature_functions = [
        get_positional_features_central_mask,
    ]

    num_components = len(feature_functions) * 2

    if (feature_size % num_components) != 0:
        raise ValueError(
            f"feature size is not divisible by number of components ({num_components})"
        )

    num_basis_per_class = feature_size // num_components

    embeddings = []
    for fn in feature_functions:
        embeddings.append(fn(distances, num_basis_per_class, seq_len))

    embeddings = torch.cat(embeddings, dim=-1)
    embeddings = torch.cat(
        (embeddings, torch.sign(distances)[..., None] * embeddings), dim=-1
    )
    return embeddings


def fast_relative_shift(a, b):
    return (
        einsum("i d, j d -> i j", a, b)
        .flatten()
        .as_strided(
            size=(a.shape[0], a.shape[0]),
            stride=((a.shape[0] - 1) * 2, 1),
            storage_offset=a.shape[0] - 1,
        )
    )


fast_relative_shift = torch.vmap(
    torch.vmap(fast_relative_shift), in_dims=(0, None)
)  # https://johahi.github.io/blog/2024/fast-relative-shift/


class Attention(nn.Module):
    def __init__(
        self,
        dim=1536,
        *,
        num_rel_pos_features=1,
        heads=8,
        dim_key=64,
        dim_value=64,
        dropout=0.0,
        pos_dropout=0.0,
    ):
        super().__init__()
        self.scale = dim_key**-0.5
        self.heads = heads

        self.to_q = nn.Linear(dim, dim_key * heads, bias=False)
        self.to_k = nn.Linear(dim, dim_key * heads, bias=False)
        self.to_v = nn.Linear(dim, dim_value * heads, bias=False)

        self.to_out = nn.Linear(dim_value * heads, dim)
        nn.init.zeros_(self.to_out.weight)
        nn.init.zeros_(self.to_out.bias)

        # relative positional encoding

        self.num_rel_pos_features = num_rel_pos_features

        self.register_buffer(
            "positions",
            get_positional_embed(
                4096, self.num_rel_pos_features, self.to_v.weight.device
            ),
            persistent=False,
        )  # 4096 as this should always be the seq len at this pos?

        self.to_rel_k = nn.Linear(num_rel_pos_features, dim_key * heads, bias=False)
        self.rel_content_bias = nn.Parameter(torch.randn(1, heads, 1, dim_key))
        self.rel_pos_bias = nn.Parameter(torch.randn(1, heads, 1, dim_key))

        # dropouts

        self.pos_dropout = nn.Dropout(pos_dropout)
        self.attn_dropout = nn.Dropout(dropout)

    def forward(self, x):
        n, h, device = x.shape[-2], self.heads, x.device

        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), (q, k, v))

        q = q * self.scale

        content_logits = einsum(
            "b h i d, b h j d -> b h i j", q + self.rel_content_bias, k
        )

        positions = self.pos_dropout(self.positions)
        rel_k = self.to_rel_k(positions)
        rel_k = rearrange(rel_k, "n (h d) -> h n d", h=h)
        rel_logits = fast_relative_shift(q + self.rel_pos_bias, rel_k)
        logits = content_logits + rel_logits
        attn = logits.softmax(dim=-1)
        attn = self.attn_dropout(attn)

        out = einsum("b h i j, b h j d -> b h i d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.to_out(out)
        return out


class _RotaryEmbedding(nn.Module):
    """NeoX-style partial rotary embedding.

    Matches the layout of ``flash_attn.layers.rotary.RotaryEmbedding``: rotates
    the first / second halves of the leading ``dim`` features of the last axis
    (``interleaved=False``). The cos/sin tables are cached as non-persistent
    buffers, so they don't appear in ``state_dict`` and don't conflict with
    checkpoints saved by the old flash-attn implementation.
    """

    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.base = float(base)
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._seq_len_cached = 0
        self._cos_cached: torch.Tensor | None = None
        self._sin_cached: torch.Tensor | None = None

    def _update_cache(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> None:
        if (
            self._cos_cached is None
            or seq_len > self._seq_len_cached
            or self._cos_cached.device != device
            or self._cos_cached.dtype != dtype
        ):
            self._seq_len_cached = seq_len
            t = torch.arange(seq_len, device=device, dtype=torch.float32)
            inv_freq = self.inv_freq.to(device=device, dtype=torch.float32)
            freqs = torch.outer(t, inv_freq)
            self._cos_cached = torch.cos(freqs).to(dtype)
            self._sin_cached = torch.sin(freqs).to(dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply rotary to the first ``self.dim`` features of ``x``.

        ``x`` has shape ``(batch, seq_len, num_heads, head_dim)``; only the
        leading ``self.dim`` of ``head_dim`` is rotated, the rest is passed
        through untouched.
        """
        seq_len = x.shape[1]
        self._update_cache(seq_len, x.device, x.dtype)
        cos = self._cos_cached[:seq_len].unsqueeze(-2)  # (seq_len, 1, dim/2)
        sin = self._sin_cached[:seq_len].unsqueeze(-2)

        rot, passthrough = x[..., : self.dim], x[..., self.dim :]
        half = self.dim // 2
        x1, x2 = rot[..., :half], rot[..., half:]
        rotated = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
        return torch.cat([rotated, passthrough], dim=-1)


class _MHAState(nn.Module):
    """Plain container so submodules show up as ``mha.Wqkv`` / ``mha.out_proj``.

    Mirrors the parameter naming of the old ``flash_attn.modules.mha.MHA`` wrapper
    so existing checkpoints (e.g. ``tom-ellis-lab/yorzoi`` on HF) load without
    any key renaming.
    """


class FlashAttention(nn.Module):
    """Multi-head attention with rotary embeddings and grouped-query attention.

    Uses PyTorch's built-in ``scaled_dot_product_attention`` (which dispatches
    to the FlashAttention 2 kernel on supported GPUs) so the model no longer
    depends on the external ``flash-attn`` package.

    The submodule layout (``self.mha.Wqkv``, ``self.mha.out_proj``) is kept
    identical to the previous flash-attn-based implementation so saved
    checkpoints continue to load.
    """

    def __init__(
        self,
        dim: int = 1536,
        heads: int = 8,
        dropout: float = 0.15,
        pos_dropout: float = 0.15,  # unused, kept for backwards compatibility
        rotary_emb_base: float = 20000.0,
        rotary_emb_scale_base=None,  # unused, kept for backwards compatibility
        rotary_emb_dim: int = 128,
    ):
        super().__init__()
        del pos_dropout, rotary_emb_scale_base  # accepted for backwards compat only

        head_dim = dim // heads
        if head_dim * heads != dim:
            raise ValueError(f"dim ({dim}) must be divisible by heads ({heads})")
        if rotary_emb_dim > head_dim:
            raise ValueError(
                f"rotary_emb_dim ({rotary_emb_dim}) must be <= head_dim ({head_dim})"
            )

        heads_kv = heads // 2
        self.heads = heads
        self.heads_kv = heads_kv
        self.head_dim = head_dim
        self.rotary_emb_dim = rotary_emb_dim
        self.softmax_scale = head_dim**-0.5
        self.dropout_p = dropout

        qkv_out_features = (heads + 2 * heads_kv) * head_dim
        self.mha = _MHAState()
        self.mha.Wqkv = nn.Linear(dim, qkv_out_features, bias=True)
        self.mha.out_proj = nn.Linear(heads * head_dim, dim, bias=True)
        self.mha.rotary_emb = _RotaryEmbedding(rotary_emb_dim, base=rotary_emb_base)

        nn.init.kaiming_normal_(self.mha.Wqkv.weight, nonlinearity="relu")
        nn.init.zeros_(self.mha.out_proj.weight)
        nn.init.zeros_(self.mha.out_proj.bias)
        nn.init.ones_(self.mha.Wqkv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        qkv = self.mha.Wqkv(x)

        q_size = self.heads * self.head_dim
        kv_size = self.heads_kv * self.head_dim
        q = qkv[..., :q_size].reshape(batch, seq_len, self.heads, self.head_dim)
        k = qkv[..., q_size : q_size + kv_size].reshape(
            batch, seq_len, self.heads_kv, self.head_dim
        )
        v = qkv[..., q_size + kv_size :].reshape(
            batch, seq_len, self.heads_kv, self.head_dim
        )

        # Rotary embedding only on Q and K (V is untouched, matching flash-attn).
        q = self.mha.rotary_emb(q)
        k = self.mha.rotary_emb(k)

        # SDPA expects (batch, num_heads, seq_len, head_dim).
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=False,
            scale=self.softmax_scale,
            enable_gqa=True,
        )

        out = out.transpose(1, 2).reshape(batch, seq_len, self.heads * self.head_dim)
        return self.mha.out_proj(out)
