"""Model factory for the v2 training engine using HybridPoseNet architecture."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig, TrainingConfig
from .geometry import rotation_6d_to_matrix


def _knn_indices(x: torch.Tensor, k: int) -> torch.Tensor:
    """Compute batched k-NN indices in feature space."""
    B, N, _ = x.shape
    k = min(k, N)
    xx = (x ** 2).sum(dim=-1, keepdim=True)
    dist = xx + xx.transpose(1, 2) - 2.0 * torch.bmm(x, x.transpose(1, 2))
    return dist.topk(k=k, dim=-1, largest=False).indices


def _gather_neighbors(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Gather neighbour features for each point."""
    B, N, C = x.shape
    k = idx.shape[-1]
    idx_base = torch.arange(B, device=x.device).view(B, 1, 1) * N
    flat_idx = (idx + idx_base).reshape(-1)
    flat_x = x.reshape(B * N, C)
    neigh = flat_x[flat_idx].reshape(B, N, k, C)
    return neigh


def _edge_features(x: torch.Tensor, k: int) -> torch.Tensor:
    """Build EdgeConv input tensor: h([x_i, x_j - x_i])."""
    idx = _knn_indices(x, k)
    neigh = _gather_neighbors(x, idx)
    center = x.unsqueeze(2).expand_as(neigh)
    edge = torch.cat([center, neigh - center], dim=-1)
    return edge.permute(0, 3, 1, 2).contiguous()


class ConvBNReLU(nn.Module):
    """Conv2d → BatchNorm → ReLU."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.bn(self.conv(x)))


class ResBlock(nn.Module):
    """Two ConvBNReLU layers with residual skip connection."""

    def __init__(self, channels: int):
        super().__init__()
        self.body = nn.Sequential(
            ConvBNReLU(channels, channels),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(x + self.body(x))


class SceneEncoder(nn.Module):
    """Shallow CNN encoder for single-channel depth image."""

    def __init__(self, base_channels: int = 32, num_blocks: int = 3, res_blocks_per_stride: int = 1, feature_dim: int = 128):
        super().__init__()
        self.stem = ConvBNReLU(1, base_channels, stride=1)
        blocks = []
        in_ch = base_channels
        for _ in range(num_blocks):
            out_ch = min(in_ch * 2, feature_dim)
            blocks.append(ConvBNReLU(in_ch, out_ch, stride=2))
            for _ in range(res_blocks_per_stride):
                blocks.append(ResBlock(out_ch))
            in_ch = out_ch
        self.blocks = nn.Sequential(*blocks)
        self.proj = nn.Conv2d(in_ch, feature_dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.blocks(x)
        x = self.proj(x)
        B, C, H, W = x.shape
        tokens = x.permute(0, 2, 3, 1).reshape(B, H * W, C)
        return tokens


class EdgeConvBlock(nn.Module):
    """EdgeConv block with dynamic k-NN graph recomputation."""

    def __init__(self, in_channels: int, out_channels: int, k: int):
        super().__init__()
        self.k = k
        self.edge_mlp = nn.Sequential(
            nn.Conv2d(in_channels * 2, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        edge = _edge_features(x, self.k)
        feat = self.edge_mlp(edge)
        feat = feat.max(dim=-1).values
        return feat.transpose(1, 2).contiguous()


class DGCNNEncoder(nn.Module):
    """Dynamic Graph CNN point encoder (Wang et al., TOG 2019)."""

    def __init__(self, edge_dims: tuple = (64, 64, 128, 256), feature_dim: int = 128, k: int = 32):
        super().__init__()
        blocks = []
        in_ch = 3
        for out_ch in edge_dims:
            blocks.append(EdgeConvBlock(in_ch, out_ch, k=k))
            in_ch = out_ch
        self.blocks = nn.ModuleList(blocks)
        total_ch = sum(edge_dims)
        self.proj = nn.Sequential(
            nn.Linear(total_ch, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU(),
        )

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        x = points
        multi_scale = []
        for block in self.blocks:
            x = block(x)
            multi_scale.append(x)
        x = torch.cat(multi_scale, dim=-1)
        return self.proj(x)


class CrossAttentionLayer(nn.Module):
    """Bidirectional cross-attention layer."""

    def __init__(self, dim: int, num_heads: int, ffn_multiplier: int = 4):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        self.norm_s1 = nn.LayerNorm(dim)
        self.norm_m1 = nn.LayerNorm(dim)
        self.q_s = nn.Linear(dim, dim)
        self.k_m = nn.Linear(dim, dim)
        self.v_m = nn.Linear(dim, dim)
        self.out_s = nn.Linear(dim, dim)

        self.norm_m2 = nn.LayerNorm(dim)
        self.norm_s2 = nn.LayerNorm(dim)
        self.q_m = nn.Linear(dim, dim)
        self.k_s = nn.Linear(dim, dim)
        self.v_s = nn.Linear(dim, dim)
        self.out_m = nn.Linear(dim, dim)

        hidden_dim = dim * ffn_multiplier
        self.norm_s_ffn = nn.LayerNorm(dim)
        self.ffn_scene = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
        self.norm_m_ffn = nn.LayerNorm(dim)
        self.ffn_model = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def _attn(self, q_seq, kv_seq, q_norm, kv_norm, q_lin, k_lin, v_lin, out_lin):
        B, Nq, C = q_seq.shape
        Nk = kv_seq.shape[1]
        H, D = self.num_heads, self.head_dim
        q = q_lin(q_norm(q_seq)).reshape(B, Nq, H, D).transpose(1, 2)
        k = k_lin(kv_norm(kv_seq)).reshape(B, Nk, H, D).transpose(1, 2)
        v = v_lin(kv_norm(kv_seq)).reshape(B, Nk, H, D).transpose(1, 2)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, Nq, C)
        return out_lin(out)

    def forward(self, scene, model):
        scene = scene + self._attn(
            scene, model, self.norm_s1, self.norm_m1, self.q_s, self.k_m, self.v_m, self.out_s
        )
        scene = scene + self.ffn_scene(self.norm_s_ffn(scene))
        model = model + self._attn(
            model, scene, self.norm_m2, self.norm_s2, self.q_m, self.k_s, self.v_s, self.out_m
        )
        model = model + self.ffn_model(self.norm_m_ffn(model))
        return scene, model


class AttentionPool(nn.Module):
    """Attention-based pooling: learnable query attends over tokens."""

    def __init__(self, dim: int):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.scale = dim**-0.5
        self.k = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)
        self.norm = nn.LayerNorm(dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        B = tokens.shape[0]
        t = self.norm(tokens)
        q = self.query.expand(B, -1, -1)
        k = self.k(t)
        v = self.v(t)
        w = (q @ k.transpose(-2, -1)) * self.scale
        w = w.softmax(dim=-1)
        return (w @ v).squeeze(1)


class HybridPool(nn.Module):
    """Combines mean, max, and attention pooling, projects 3C → C."""

    def __init__(self, dim: int):
        super().__init__()
        self.attn_pool = AttentionPool(dim)
        self.proj = nn.Linear(dim * 3, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        mean_f = tokens.mean(dim=1)
        max_f = tokens.max(dim=1).values
        attn_f = self.attn_pool(tokens)
        fused = torch.cat([mean_f, max_f, attn_f], dim=-1)
        return self.norm(F.gelu(self.proj(fused)))


class PoseHead(nn.Module):
    """MLP that maps fused features to translation, 6D rotation, and confidence scores."""

    def __init__(self, in_features: int, hidden_dims: list, dropout: float = 0.3):
        super().__init__()

        def _build_branch(input_dim):
            layers = []
            prev = input_dim
            for dim in hidden_dims:
                layers.extend(
                    [
                        nn.Linear(prev, dim),
                        nn.LayerNorm(dim),
                        nn.ReLU(),
                        nn.Dropout(dropout),
                    ]
                )
                prev = dim
            return nn.Sequential(*layers), prev

        self.trans_mlp, trans_out_dim = _build_branch(in_features)
        self.rot_mlp, rot_out_dim = _build_branch(in_features)
        self.translation_head = nn.Linear(trans_out_dim, 3)
        self.rotation_head = nn.Linear(rot_out_dim, 6)
        # Confidence heads: predict scalar confidence in [0, 1] for each modality
        self.confidence_t_head = nn.Linear(trans_out_dim, 1)
        self.confidence_r_head = nn.Linear(rot_out_dim, 1)

    def forward(self, x: torch.Tensor):
        trans_feat = self.trans_mlp(x)
        rot_feat = self.rot_mlp(x)
        translation = self.translation_head(trans_feat)
        rotation = rotation_6d_to_matrix(self.rotation_head(rot_feat))
        # Confidence scores bounded to [0, 1]
        confidence_t = torch.sigmoid(self.confidence_t_head(trans_feat)).squeeze(-1)  # [B]
        confidence_r = torch.sigmoid(self.confidence_r_head(rot_feat)).squeeze(-1)  # [B]
        return translation, rotation, confidence_t, confidence_r


class HybridPoseNet(nn.Module):
    """Hybrid CNN + DGCNN pose estimation network."""

    def __init__(self, config: ModelConfig, camera_config: dict | None = None):
        super().__init__()
        camera_config = camera_config or {}
        se_cfg = config.scene_encoder
        me_cfg = config.point_encoder
        feat_dim = se_cfg.feature_dim

        self.scene_encoder = SceneEncoder(
            base_channels=se_cfg.base_channels,
            num_blocks=se_cfg.num_blocks,
            res_blocks_per_stride=se_cfg.res_blocks_per_stride,
            feature_dim=feat_dim,
        )
        self.model_encoder = DGCNNEncoder(
            edge_dims=tuple(me_cfg.hidden_dims),
            feature_dim=feat_dim,
            k=me_cfg.k,
        )

        in_h = int(camera_config.get("padded_height", 256))
        in_w = int(camera_config.get("resolution", {}).get("width", 320))
        num_blocks = int(se_cfg.num_blocks)
        token_h = in_h // (2**num_blocks)
        token_w = in_w // (2**num_blocks)
        scene_pos = self._build_2d_sincos_pos_encoding(token_h, token_w, feat_dim)
        self.register_buffer("scene_pos_2d", scene_pos.unsqueeze(0), persistent=True)

        self.cross_attention = nn.ModuleList(
            [CrossAttentionLayer(feat_dim, num_heads=config.cross_attention.num_heads)
             for _ in range(config.cross_attention.num_layers)]
        )
        self.model_coord_proj = nn.Sequential(
            nn.Linear(3, feat_dim),
            nn.LayerNorm(feat_dim),
        )
        self.scene_pool = HybridPool(feat_dim)
        self.model_pool = HybridPool(feat_dim)
        self.pose_head = PoseHead(
            in_features=feat_dim * 2,
            hidden_dims=config.fusion.hidden_dims,
            dropout=config.fusion.dropout,
        )

    @staticmethod
    def _build_2d_sincos_pos_encoding(height: int, width: int, dim: int) -> torch.Tensor:
        """Create flattened 2D sine/cosine encoding."""
        y, x = torch.meshgrid(
            torch.linspace(0.0, 1.0, height, dtype=torch.float32),
            torch.linspace(0.0, 1.0, width, dtype=torch.float32),
            indexing="ij",
        )
        pos_x = x.reshape(-1, 1)
        pos_y = y.reshape(-1, 1)
        num_freq = max(dim // 4, 1)
        freq_idx = torch.arange(num_freq, dtype=torch.float32)
        denom = max(num_freq - 1, 1)
        omega = 1.0 / (10000.0 ** (freq_idx / denom))
        omega = omega.unsqueeze(0)
        phase_x = 2.0 * math.pi * pos_x * omega
        phase_y = 2.0 * math.pi * pos_y * omega
        enc = torch.cat(
            [
                torch.sin(phase_x),
                torch.cos(phase_x),
                torch.sin(phase_y),
                torch.cos(phase_y),
            ],
            dim=-1,
        )
        if enc.shape[-1] < dim:
            enc = F.pad(enc, (0, dim - enc.shape[-1]))
        return enc[:, :dim]

    def forward(self, depth: torch.Tensor, model_points: torch.Tensor):
        scene_tokens = self.scene_encoder(depth)
        model_tokens = self.model_encoder(model_points)
        scene_pos = self.scene_pos_2d.to(device=scene_tokens.device, dtype=scene_tokens.dtype)
        scene_tokens = scene_tokens + scene_pos
        model_pos = self.model_coord_proj(model_points.to(model_tokens.dtype))
        model_tokens = model_tokens + model_pos
        for layer in self.cross_attention:
            scene_tokens, model_tokens = layer(scene_tokens, model_tokens)
        scene_global = self.scene_pool(scene_tokens)
        model_global = self.model_pool(model_tokens)
        fused = torch.cat([scene_global, model_global], dim=-1)
        translation, rotation, confidence_t, confidence_r = self.pose_head(fused)
        return {
            "translation": translation,
            "rotation_6d": None,
            "pred_t": translation,
            "pred_R": rotation,
            "confidence_t": confidence_t,
            "confidence_r": confidence_r,
        }


def build_model(config: TrainingConfig) -> HybridPoseNet:
    camera_config = {
        "padded_height": 256,
        "resolution": {"width": 320, "height": 240},
    }
    return HybridPoseNet(config.model, camera_config=camera_config)