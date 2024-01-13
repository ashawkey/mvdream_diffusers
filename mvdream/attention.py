import torch
import torch.nn as nn
import torch.nn.functional as F

from inspect import isfunction
from einops import rearrange, repeat
from typing import Optional, Any

# require xformers
import xformers  # type: ignore
import xformers.ops  # type: ignore

from .util import checkpoint, zero_module

def default(val, d):
    if val is not None:
        return val
    return d() if isfunction(d) else d


class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.0):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = (
            nn.Sequential(nn.Linear(dim, inner_dim), nn.GELU())
            if not glu
            else GEGLU(dim, inner_dim)
        )

        self.net = nn.Sequential(
            project_in, nn.Dropout(dropout), nn.Linear(inner_dim, dim_out)
        )

    def forward(self, x):
        return self.net(x)


class MemoryEfficientCrossAttention(nn.Module):
    # https://github.com/MatthieuTPHR/diffusers/blob/d80b531ff8060ec1ea982b65a1b8df70f73aa67c/src/diffusers/models/attention.py#L223
    def __init__(
            self, 
            query_dim, 
            context_dim=None, 
            heads=8, 
            dim_head=64, 
            dropout=0.0,
            ip_dim=0,
            ip_weight=1,
        ):
        super().__init__()
        
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.heads = heads
        self.dim_head = dim_head

        self.ip_dim = ip_dim
        self.ip_weight = ip_weight

        if self.ip_dim > 0:
            self.to_k_ip = nn.Linear(context_dim, inner_dim, bias=False)
            self.to_v_ip = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim), nn.Dropout(dropout)
        )
        self.attention_op: Optional[Any] = None

    def forward(self, x, context=None):
        q = self.to_q(x)
        context = default(context, x)

        if self.ip_dim > 0:
            # context： [B, 77 + 16(ip), 1024]
            token_len = context.shape[1]
            context_ip = context[:, -self.ip_dim :, :]
            k_ip = self.to_k_ip(context_ip)
            v_ip = self.to_v_ip(context_ip)
            context = context[:, : (token_len - self.ip_dim), :]

        k = self.to_k(context)
        v = self.to_v(context)

        b, _, _ = q.shape
        q, k, v = map(
            lambda t: t.unsqueeze(3)
            .reshape(b, t.shape[1], self.heads, self.dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b * self.heads, t.shape[1], self.dim_head)
            .contiguous(),
            (q, k, v),
        )

        # actually compute the attention, what we cannot get enough of
        out = xformers.ops.memory_efficient_attention(
            q, k, v, attn_bias=None, op=self.attention_op
        )

        if self.ip_dim > 0:
            k_ip, v_ip = map(
                lambda t: t.unsqueeze(3)
                .reshape(b, t.shape[1], self.heads, self.dim_head)
                .permute(0, 2, 1, 3)
                .reshape(b * self.heads, t.shape[1], self.dim_head)
                .contiguous(),
                (k_ip, v_ip),
            )
            # actually compute the attention, what we cannot get enough of
            out_ip = xformers.ops.memory_efficient_attention(
                q, k_ip, v_ip, attn_bias=None, op=self.attention_op
            )
            out = out + self.ip_weight * out_ip

        out = (
            out.unsqueeze(0)
            .reshape(b, self.heads, out.shape[1], self.dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b, out.shape[1], self.heads * self.dim_head)
        )
        return self.to_out(out)


class BasicTransformerBlock3D(nn.Module):
    
    def __init__(
        self,
        dim,
        n_heads,
        d_head,
        context_dim,
        dropout=0.0,
        gated_ff=True,
        checkpoint=True,
        ip_dim=0,
        ip_weight=1,
    ):
        super().__init__()

        self.attn1 = MemoryEfficientCrossAttention(
            query_dim=dim,
            context_dim=None, # self-attention
            heads=n_heads,
            dim_head=d_head,
            dropout=dropout,
        )
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.attn2 = MemoryEfficientCrossAttention(
            query_dim=dim,
            context_dim=context_dim,
            heads=n_heads,
            dim_head=d_head,
            dropout=dropout,
            # ip only applies to cross-attention
            ip_dim=ip_dim,
            ip_weight=ip_weight,
        ) 
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.checkpoint = checkpoint

    def forward(self, x, context=None, num_frames=1):
        return checkpoint(
            self._forward, (x, context, num_frames), self.parameters(), self.checkpoint
        )

    def _forward(self, x, context=None, num_frames=1):
        x = rearrange(x, "(b f) l c -> b (f l) c", f=num_frames).contiguous()
        x = self.attn1(self.norm1(x), context=None) + x
        x = rearrange(x, "b (f l) c -> (b f) l c", f=num_frames).contiguous()
        x = self.attn2(self.norm2(x), context=context) + x
        x = self.ff(self.norm3(x)) + x
        return x


class SpatialTransformer3D(nn.Module):

    def __init__(
        self,
        in_channels,
        n_heads,
        d_head,
        context_dim, # cross attention input dim
        depth=1,
        dropout=0.0,
        ip_dim=0,
        ip_weight=1,
        use_checkpoint=True,
    ):
        super().__init__()

        if not isinstance(context_dim, list):
            context_dim = [context_dim]

        self.in_channels = in_channels

        inner_dim = n_heads * d_head
        self.norm = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
        self.proj_in = nn.Linear(in_channels, inner_dim)

        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock3D(
                    inner_dim,
                    n_heads,
                    d_head,
                    context_dim=context_dim[d],
                    dropout=dropout,
                    checkpoint=use_checkpoint,
                    ip_dim=ip_dim,
                    ip_weight=ip_weight,
                )
                for d in range(depth)
            ]
        )
        
        self.proj_out = zero_module(nn.Linear(in_channels, inner_dim))
        

    def forward(self, x, context=None, num_frames=1):
        # note: if no context is given, cross-attention defaults to self-attention
        if not isinstance(context, list):
            context = [context]
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)
        x = rearrange(x, "b c h w -> b (h w) c").contiguous()
        x = self.proj_in(x)
        for i, block in enumerate(self.transformer_blocks):
            x = block(x, context=context[i], num_frames=num_frames)
        x = self.proj_out(x)
        x = rearrange(x, "b (h w) c -> b c h w", h=h, w=w).contiguous()
        
        return x + x_in
