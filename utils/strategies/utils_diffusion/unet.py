import torch
import torch.nn as nn
import math
from typing import Optional

# -------------------------
# 时间正弦编码
# -------------------------
def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000):
    if t.dim() == 2 and t.shape[1] == 1:
        t = t[:, 0]
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(0, half, dtype=torch.float32, device=t.device) / half
    )
    args = t[:, None].float() * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb

# -------------------------
# 改进后的 ResidualBlockFiLM
# -------------------------
class ResidualBlockFiLM(nn.Module):
    def __init__(self, dim, hidden, time_emb_dim: Optional[int] = None,
                 dropout: float = 0.0, norm: bool = True):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim) if norm else nn.Identity()
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.norm2 = nn.LayerNorm(hidden) if norm else nn.Identity()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden, dim)

        if time_emb_dim is not None:
            self.time_proj = nn.Linear(time_emb_dim, hidden * 2)
            # 关键优化：将 FiLM 的投影层权重和偏置初始化为 0
            # 这样初始状态下 gamma=0, beta=0，block 对时间不敏感，利于初期训练
            nn.init.zeros_(self.time_proj.weight)
            nn.init.zeros_(self.time_proj.bias)
        else:
            self.time_proj = None

    def forward(self, x, t_emb: Optional[torch.Tensor] = None):
        resid = x
        h = self.norm1(x)
        h = self.fc1(h)
        h = self.act(h)

        if (self.time_proj is not None) and (t_emb is not None):
            film = self.time_proj(t_emb)
            gamma, beta = film.chunk(2, dim=-1)
            h = h * (1 + gamma) + beta

        h = self.norm2(h)
        h = self.dropout(h)
        h = self.fc2(h)
        return resid + h

# -------------------------
# Time MLP
# -------------------------
class TimeMLP(nn.Module):
    def __init__(self, time_dim: int, out_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(time_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, out_dim)
        )
    def forward(self, t):
        return self.net(t)

# -------------------------
# 优化后的 TinyUNet1D
# -------------------------
class TinyUNet1D(nn.Module):
    def __init__(
        self,
        dim: int = 32,
        hidden: int = 256,
        depth: int = 6,               # 几个 ResidualBlockFiLM
        time_embed_dim: int = 128,
        dropout: float = 0.0,
        use_skip: bool = True,        # 是否保留可选 skip 机制
    ):
        super().__init__()
        self.time_embed_dim = time_embed_dim
        self.use_skip = use_skip
        # 内部真正的通道维度：仅基于 x 的 hidden
        self.inner_dim = hidden

        # 输入层：将 [x, cond] 拼接后的向量 (2 * dim) 投影到 hidden
        self.input_proj_x = nn.Linear(2 * dim, hidden)

        self.act = nn.GELU()

        # 时间 MLP，直接输出 inner_dim 以做 FiLM
        self.time_mlp = TimeMLP(time_embed_dim, out_dim=self.inner_dim, hidden=self.inner_dim)

        # 一串 ResidualBlockFiLM = 精简版 UNet
        self.blocks = nn.ModuleList([
            ResidualBlockFiLM(self.inner_dim, self.inner_dim, time_emb_dim=self.inner_dim, dropout=dropout)
            for _ in range(depth)
        ])

        # skip 连接的压缩层（输入为 [h, skip] 拼接 -> 2 * inner_dim）
        if use_skip:
            self.skip_proj = nn.ModuleList([
                nn.Linear(2 * self.inner_dim, self.inner_dim) for _ in range(depth)
            ])

        self.out_norm = nn.LayerNorm(self.inner_dim)
        self.out_layer = nn.Linear(self.inner_dim, dim)

    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor):
        """
        x:    [B, D]
        cond: [B, D] 与 x 维度一致，作为条件
        t:    [B]
        """
        B = x.shape[0]
        t = t.reshape(B)

        # --- 时间嵌入 ---
        t_sin = timestep_embedding(t, self.time_embed_dim)
        t_feat = self.time_mlp(t_sin)

        # --- 输入拼接与投影 ---
        h_in = torch.cat([x, cond], dim=-1)  # [B, 2 * D]
        h = self.input_proj_x(h_in)
        h = self.act(h)

        skips = []

        # --- 主网络（相当于 UNet 的 trunk）---
        for i, blk in enumerate(self.blocks):
            h = blk(h, t_feat)
            if self.use_skip:
                skips.append(h)

        # --- 反向融合（简化版 skip）---
        if self.use_skip:
            for proj, skip in zip(self.skip_proj[::-1], skips[::-1]):
                h = torch.cat([h, skip], dim=-1)  # [B, 2 * inner_dim]
                h = proj(h)                       # [B, inner_dim]

        # --- 输出层 ---
        h = self.out_norm(h)
        out = self.out_layer(h)
        return out
    

class TimeConditionalClassifier(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden: int = 256,
        depth: int = 6,
        time_embed_dim: int = 128,
        dropout: float = 0.0,
        use_skip: bool = True,
    ):
        super().__init__()
        self.time_embed_dim = time_embed_dim
        self.use_skip = use_skip
        self.inner_dim = hidden

        # 输入：[x, cond] -> hidden
        self.input_proj_x = nn.Linear(dim, hidden)
        self.act = nn.GELU()

        # 时间 MLP，与 TinyUNet1D 一致
        self.time_mlp = TimeMLP(time_embed_dim, out_dim=self.inner_dim, hidden=self.inner_dim)

        # ResidualBlockFiLM 堆叠
        self.blocks = nn.ModuleList([
            ResidualBlockFiLM(self.inner_dim, self.inner_dim, time_emb_dim=self.inner_dim, dropout=dropout)
            for _ in range(depth)
        ])

        # 可选 skip 连接
        if use_skip:
            self.skip_proj = nn.ModuleList([
                nn.Linear(2 * self.inner_dim, self.inner_dim) for _ in range(depth)
            ])

        self.out_norm = nn.LayerNorm(self.inner_dim)
        # 输出标量 logits
        self.out_layer = nn.Linear(self.inner_dim, 1)

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        """
        x:    [B, D]
        cond: [B, D] 与 x 维度一致，作为条件
        t:    [B]
        """
        B = x.shape[0]
        t = t.reshape(B)

        # 时间嵌入
        t_sin = timestep_embedding(t, self.time_embed_dim)
        t_feat = self.time_mlp(t_sin)

        # 输入拼接与投影
        # h_in = torch.cat([x, cond], dim=-1)  # [B, 2 * D]
        h_in = x
        h = self.input_proj_x(h_in)
        h = self.act(h)

        skips = []

        # 主干网络
        for blk in self.blocks:
            h = blk(h, t_feat)
            if self.use_skip:
                skips.append(h)

        # 反向融合 skip
        if self.use_skip:
            for proj, skip in zip(self.skip_proj[::-1], skips[::-1]):
                h = torch.cat([h, skip], dim=-1)  # [B, 2 * inner_dim]
                h = proj(h)                       # [B, inner_dim]

        # 输出层
        h = self.out_norm(h)
        logits = self.out_layer(h)  # [B, 1]
        return logits