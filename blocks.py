"""
Lamaakal, I., Yahyati, C., Maleh, Y. et al. 
A tiny inertial transformer for human activity recognition via multimodal knowledge 
distillation and explainable AI. Sci Rep 15, 42335 (2025). 
https://doi.org/10.1038/s41598-025-26297-2

MIT License

Copyright (c) 2025 Ismail Lamaakal

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

https://github.com/Ism-ail11/XTinyHAR/tree/main?tab=MIT-1-ov-file
"""

import torch
import torch.nn as nn
from torch import Tensor

class PositionalEmbeddingLearned(nn.Module):
    def __init__(self, seq_len: int, dim: int):
        super().__init__()
        self.pos = nn.Parameter(torch.zeros(1, seq_len, dim))
        nn.init.trunc_normal_(self.pos, std=0.02)

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, N, D)
        return x + self.pos[:, : x.size(1), :]

class MLP(nn.Module):
    def __init__(self, dim, hidden_dim, drop=0.1):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class MHSA(nn.Module):
    def __init__(self, dim, num_heads=4, attn_drop=0.1, proj_drop=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert dim % num_heads == 0, "dim must be divisible by heads"
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: Tensor):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2,0,3,1,4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B, heads, N, head_dim)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        out = attn @ v  # (B, heads, N, head_dim)
        out = out.transpose(1,2).reshape(B, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out, attn  # return attention for XAI if needed

class TransformerEncoderLayer(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=2.0, drop=0.1, attn_drop=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MHSA(dim, num_heads, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio), drop=drop)

    def forward(self, x: Tensor):
        attn_in = self.norm1(x)
        attn_out, attn_map = self.attn(attn_in)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x, attn_map

class PatchEmbed1D(nn.Module):
    """
    Flattens non-overlapping temporal patches of size P across C channels.
    Input: (B, W, C) → Output: (B, N, D) with N = floor(W/P)
    """
    def __init__(self, in_channels: int, patch_size: int, embed_dim: int):
        super().__init__()
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.proj = nn.Linear(patch_size * in_channels, embed_dim)

    def forward(self, x: Tensor) -> Tensor:
        B, W, C = x.shape
        P = self.patch_size
        N = W // P
        x = x[:, :N*P, :]
        x = x.reshape(B, N, P*C)  # (B, N, P*C)
        x = self.proj(x)          # (B, N, D)
        return x