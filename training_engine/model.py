"""Model factory for the v2 training engine using HybridPoseNet architecture."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig, SegmentationConfig, TrainingConfig
from .geometry import rotation_6d_to_matrix


def _check_finite(name: str, tensor: torch.Tensor) -> None:
    if not torch.isfinite(tensor).all():
        bad_count = int((~torch.isfinite(tensor)).sum().item())
        raise RuntimeError(
            f"Non-finite tensor at {name}: dtype={tensor.dtype}, shape={tuple(tensor.shape)}, bad_count={bad_count}"
        )


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
        out = F.relu(self.bn(self.conv(x)))
        _check_finite(f"ConvBNReLU[{self.conv.in_channels}->{self.conv.out_channels}]", out)
        return out


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
        out = F.relu(x + self.body(x))
        _check_finite(f"ResBlock[{x.shape[1]}]", out)
        return out


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

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        with torch.autocast(device_type=x.device.type, enabled=False):
            x = self.stem(x)
            _check_finite("SceneEncoder.stem", x)
            # x = x.to(dtype=torch.float32)
            x = self.blocks(x)
            x = self.proj(x)
            B, C, H, W = x.shape
            tokens = x.permute(0, 2, 3, 1).reshape(B, H * W, C)
        tokens = tokens.to(dtype=torch.float16)  # cast back to fp16 for downstream layers
        _check_finite("SceneEncoder.tokens", tokens)
        return tokens, (H, W)


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
        _check_finite("EdgeConvBlock.edge", edge)
        feat = self.edge_mlp(edge)
        _check_finite("EdgeConvBlock.edge_mlp", feat)
        feat = feat.max(dim=-1).values
        feat = feat.transpose(1, 2).contiguous()
        _check_finite("EdgeConvBlock.output", feat)
        return feat


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
        _check_finite("DGCNNEncoder.concat", x)
        x = self.proj(x)
        _check_finite("DGCNNEncoder.proj", x)
        return x


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
        _check_finite("CrossAttention.q", q)
        _check_finite("CrossAttention.k", k)
        _check_finite("CrossAttention.v", v)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        _check_finite("CrossAttention.attn", attn)
        out = (attn @ v).transpose(1, 2).reshape(B, Nq, C)
        out = out_lin(out)
        _check_finite("CrossAttention.out", out)
        return out

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

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        B, N, _ = tokens.shape
        t = self.norm(tokens)
        _check_finite("AttentionPool.norm", t)
        q = self.query.expand(B, -1, -1)
        k = self.k(t)
        v = self.v(t)
        w = (q @ k.transpose(-2, -1)) * self.scale
        if mask is not None:
            w = w.masked_fill(mask.unsqueeze(1) < 0.5, -1e4)
        w = w.softmax(dim=-1)
        if mask is not None:
            invalid = (mask > 0.5).sum(dim=1) < 1
            if invalid.any():
                w = torch.where(
                    invalid.view(B, 1, 1),
                    torch.full_like(w, 1.0 / max(N, 1)),
                    w,
                )
        out = (w @ v).squeeze(1)
        _check_finite("AttentionPool.out", out)
        return out


class HybridPool(nn.Module):
    """Combines mean, max, and attention pooling, projects 3C → C.

    Optional ``mask`` [B, N] in [0, 1] reweights scene tokens (masked mean / max / attention).
    """

    def __init__(self, dim: int):
        super().__init__()
        self.attn_pool = AttentionPool(dim)
        self.proj = nn.Linear(dim * 3, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if mask is None:
            mean_f = tokens.mean(dim=1)
            max_f = tokens.max(dim=1).values
            attn_f = self.attn_pool(tokens)
        else:
            m = mask.clamp(0.0, 1.0).unsqueeze(-1)
            wsum = m.sum(dim=1).clamp(min=1e-6)
            mean_f = (tokens * m).sum(dim=1) / wsum
            neg = torch.tensor(-1e4, device=tokens.device, dtype=torch.float32)
            tok32 = tokens.to(dtype=torch.float32)
            m32 = m.to(dtype=torch.float32)
            masked_max = tok32.masked_fill(m32 < 0.5, neg)
            max_f = masked_max.max(dim=1).values.to(dtype=tokens.dtype)
            attn_f = self.attn_pool(tokens, mask=mask)
        fused = torch.cat([mean_f, max_f, attn_f], dim=-1)
        _check_finite("HybridPool.fused", fused)
        with torch.autocast(device_type=fused.device.type, enabled=False):
            fused = fused.to(dtype=torch.float32)
            out = self.norm(F.gelu(self.proj(fused)))
            out = out.to(dtype=tokens.dtype)
        _check_finite("HybridPool.out", out)
        return out


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
        _check_finite("PoseHead.trans_feat", trans_feat)
        _check_finite("PoseHead.rot_feat", rot_feat)
        translation = self.translation_head(trans_feat)
        _check_finite("PoseHead.translation", translation)
        rotation_6d = self.rotation_head(rot_feat)
        _check_finite("PoseHead.rotation_6d", rotation_6d)
        rotation = rotation_6d_to_matrix(rotation_6d)
        _check_finite("PoseHead.rotation_matrix", rotation)
        # Confidence scores bounded to [0, 1]
        confidence_t = torch.sigmoid(self.confidence_t_head(trans_feat)).squeeze(-1)  # [B]
        confidence_r = torch.sigmoid(self.confidence_r_head(rot_feat)).squeeze(-1)  # [B]
        _check_finite("PoseHead.confidence_t", confidence_t)
        _check_finite("PoseHead.confidence_r", confidence_r)
        return translation, rotation, confidence_t, confidence_r


class HybridPoseNet(nn.Module):
    """Hybrid CNN + DGCNN pose estimation network."""

    def __init__(
        self,
        config: ModelConfig,
        segmentation: SegmentationConfig | None = None,
        camera_config: dict | None = None,
    ):
        super().__init__()
        _ = camera_config or {}
        se_cfg = config.scene_encoder
        me_cfg = config.point_encoder
        feat_dim = se_cfg.feature_dim

        self.segmentation = segmentation or SegmentationConfig(enabled=False)
        self.use_segmentation = bool(self.segmentation.enabled)

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

        self._pos_cache: dict[tuple[int, int], torch.Tensor] = {}

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
        self.scene_mask_head = nn.Linear(feat_dim, 1) if self.use_segmentation else None

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

    def _segmentation_pose_branch(
        self,
        scene_with_pos: torch.Tensor,
        gate_mask: torch.Tensor,
        model_tokens_init: torch.Tensor,
        *,
        tag: str,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run gated scene tokens through cross-attention, pool, and pose head (shared weights)."""
        scene_tokens = scene_with_pos * gate_mask.unsqueeze(-1)
        _check_finite(f"HybridPoseNet.scene_tokens_masked/{tag}", scene_tokens)
        model_tokens = model_tokens_init
        for layer in self.cross_attention:
            scene_tokens, model_tokens = layer(scene_tokens, model_tokens)
            _check_finite(f"HybridPoseNet.scene_tokens.attn/{tag}", scene_tokens)
            _check_finite(f"HybridPoseNet.model_tokens.attn/{tag}", model_tokens)

        model_global = self.model_pool(model_tokens, mask=None)
        scene_global = self.scene_pool(scene_tokens, mask=gate_mask)
        fused = torch.cat([scene_global, model_global], dim=-1)
        return self.pose_head(fused)

    def forward(
        self,
        depth: torch.Tensor,
        model_points: torch.Tensor,
        scene_mask: torch.Tensor | None = None,
        *,
        train: bool = False,
    ) -> dict[str, Any]:
        scene_raw, (H, W) = self.scene_encoder(depth)
        model_tokens = self.model_encoder(model_points)
        _check_finite("HybridPoseNet.scene_tokens", scene_raw)
        _check_finite("HybridPoseNet.model_tokens", model_tokens)

        key = (H, W)
        if key not in getattr(self, "_pos_cache", {}) or self._pos_cache[key].device != scene_raw.device:
            pos = self._build_2d_sincos_pos_encoding(H, W, scene_raw.shape[-1])
            if not hasattr(self, "_pos_cache"):
                self._pos_cache = {}
            self._pos_cache[key] = pos.unsqueeze(0).to(device=scene_raw.device, dtype=scene_raw.dtype)

        scene_with_pos = scene_raw + self._pos_cache[key]
        _check_finite("HybridPoseNet.scene_tokens+pos", scene_with_pos)

        model_tokens = model_tokens + self.model_coord_proj(model_points.to(model_tokens.dtype))
        _check_finite("HybridPoseNet.model_tokens+pos", model_tokens)

        if not self.use_segmentation:
            scene_tokens = scene_with_pos
            for layer in self.cross_attention:
                scene_tokens, model_tokens = layer(scene_tokens, model_tokens)
                _check_finite("HybridPoseNet.scene_tokens.attn", scene_tokens)
                _check_finite("HybridPoseNet.model_tokens.attn", model_tokens)

            model_global = self.model_pool(model_tokens, mask=None)
            scene_global = self.scene_pool(scene_tokens, mask=None)
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

        assert self.scene_mask_head is not None
        mask_logits = self.scene_mask_head(scene_raw).squeeze(-1)
        _check_finite("HybridPoseNet.mask_logits", mask_logits)
        mask_pred = torch.sigmoid(mask_logits)

        mask_gt_tok: torch.Tensor | None = None
        if scene_mask is not None:
            mask_gt_tok = (
                F.interpolate(scene_mask.float(), size=(H, W), mode="nearest")
                .reshape(scene_mask.shape[0], H * W)
                .to(dtype=scene_raw.dtype)
            )

        pred_hard = (mask_pred > 0.5).to(scene_raw.dtype)

        dual_eval = (not train) and (mask_gt_tok is not None)
        if dual_eval:
            trans_p, rot_p, ct_p, cr_p = self._segmentation_pose_branch(
                scene_with_pos, pred_hard, model_tokens.clone(), tag="pred_mask"
            )
            trans_g, rot_g, ct_g, cr_g = self._segmentation_pose_branch(
                scene_with_pos, mask_gt_tok, model_tokens.clone(), tag="gt_mask"
            )
            return {
                "translation": trans_p,
                "rotation_6d": None,
                "pred_t": trans_p,
                "pred_R": rot_p,
                "confidence_t": ct_p,
                "confidence_r": cr_p,
                "pred_t_gt_mask": trans_g,
                "pred_R_gt_mask": rot_g,
                "confidence_t_gt_mask": ct_g,
                "confidence_r_gt_mask": cr_g,
                "mask_logits": mask_logits,
                "mask_gt_tokens": mask_gt_tok,
            }

        if train and self.segmentation.train_pose_with_gt_mask and mask_gt_tok is not None:
            gate = mask_gt_tok
        else:
            gate = pred_hard

        translation, rotation, confidence_t, confidence_r = self._segmentation_pose_branch(
            scene_with_pos, gate, model_tokens, tag="train" if train else "pred_mask"
        )
        return {
            "translation": translation,
            "rotation_6d": None,
            "pred_t": translation,
            "pred_R": rotation,
            "confidence_t": confidence_t,
            "confidence_r": confidence_r,
            "mask_logits": mask_logits,
            "mask_gt_tokens": mask_gt_tok,
        }


def build_model(config: TrainingConfig) -> HybridPoseNet:
    camera_config = {
        "padded_height": 256,
        "resolution": {"width": 320, "height": 240},
    }
    return HybridPoseNet(config.model, segmentation=config.segmentation, camera_config=camera_config)