"""Pose losses for the v2 training engine."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .config import LossConfig
from .geometry import rotation_geodesic_error


@dataclass(frozen=True)
class PoseLossWeights:
    translation_weight: float = 1.0
    rotation_weight: float = 0.5
    bbox_corner_weight: float = 2.0


def _transform_bbox_corners(corners: torch.Tensor,
                             transform: torch.Tensor) -> torch.Tensor:
    """
    Apply a [B,4,4] transformation to [B,8,3] corners.
    Returns [B, 8, 3].
    """
    B, N, _ = corners.shape
    ones = torch.ones(B, N, 1, device=corners.device, dtype=corners.dtype)
    h    = torch.cat([corners, ones], dim=-1)          # [B, 8, 4]
    out  = torch.bmm(transform, h.transpose(1, 2))    # [B, 4, 8]
    return out[:, :3, :].transpose(1, 2)               # [B, 8, 3]


def _aabb_iou(corners1: torch.Tensor, corners2: torch.Tensor) -> torch.Tensor:
    """Compute AABB IoU from two [B,8,3] oriented-bbox corner sets. Returns [B]."""
    min1, max1 = corners1.min(dim=1)[0], corners1.max(dim=1)[0]  # [B, 3]
    min2, max2 = corners2.min(dim=1)[0], corners2.max(dim=1)[0]

    inter_min = torch.max(min1, min2)
    inter_max = torch.min(max1, max2)
    inter_dims = torch.clamp(inter_max - inter_min, min=0.0)
    inter_vol  = inter_dims.prod(dim=-1)                         # [B]

    vol1 = (max1 - min1).prod(dim=-1)
    vol2 = (max2 - min2).prod(dim=-1)

    return inter_vol / (vol1 + vol2 - inter_vol + 1e-8)          # [B]


def confidence_loss(
    pred_conf_t: torch.Tensor,
    pred_conf_r: torch.Tensor,
    translation_error: torch.Tensor,
    rotation_error_deg: torch.Tensor,
    temperature: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Train confidence heads as inverse of prediction error.
    
    Args:
        pred_conf_t: [B] predicted translation confidence in [0,1]
        pred_conf_r: [B] predicted rotation confidence in [0,1]
        translation_error: [B] translation L2 error in meters
        rotation_error_deg: [B] rotation geodesic error in degrees
        temperature: scaling factor for exponential decay
    
    Returns:
        dict with 'loss', 'conf_t', 'conf_r' keys
    """
    # Target: confidence = exp(-temperature * error)
    target_conf_t = torch.exp(-temperature * translation_error)
    target_conf_r = torch.exp(-temperature * (rotation_error_deg / 180.0))
    
    loss_t = F.mse_loss(pred_conf_t, target_conf_t)
    loss_r = F.mse_loss(pred_conf_r, target_conf_r)
    
    return {
        "loss": loss_t + loss_r,
        "conf_t": loss_t.detach(),
        "conf_r": loss_r.detach(),
    }


def pose_loss(
    pred_transform: torch.Tensor,
    gt_transform: torch.Tensor,
    bbox_corners: torch.Tensor,
    pred_conf_t: torch.Tensor | None = None,
    pred_conf_r: torch.Tensor | None = None,
    weights: PoseLossWeights | LossConfig | None = None,
) -> dict[str, torch.Tensor]:
    """Compute pose loss over translation, rotation, bbox corners, and confidence.
    
    Uses axis-weighted translation loss, SO(3) surrogate rotation loss,
    and 3D IoU bbox loss from the original project, plus optional confidence loss.
    
    Args:
        pred_transform: [B, 4, 4] predicted transformation matrices
        gt_transform: [B, 4, 4] ground truth transformation matrices
        bbox_corners: [B, 8, 3] canonical bbox corners in model frame
        pred_conf_t: [B] optional predicted translation confidence
        pred_conf_r: [B] optional predicted rotation confidence
        weights: PoseLossWeights or LossConfig with loss weights
    
    Returns:
        dict with loss components
    """
    if weights is None:
        weights = PoseLossWeights()
    
    # Extract weights
    trans_w = weights.translation_weight
    rot_w = weights.rotation_weight
    bbox_w = weights.bbox_corner_weight
    conf_w = getattr(weights, "confidence_weight", 0.0)
    conf_temp = getattr(weights, "confidence_temperature", 1.0)
    
    pred_t = pred_transform[:, :3, 3]           # [B, 3]
    gt_t   = gt_transform[:, :3, 3]
    pred_R = pred_transform[:, :3, :3]          # [B, 3, 3]
    gt_R   = gt_transform[:, :3, :3]

    # ── Translation loss (axis-weighted MSE) ─────────────────────────────────
    axis_weights_cfg = getattr(weights, "translation_axis_weights", [10.0, 10.0, 1.0])
    if len(axis_weights_cfg) != 3:
        raise ValueError("translation_axis_weights must contain exactly 3 values")

    axis_weights = torch.as_tensor(axis_weights_cfg,
                                   device=pred_t.device, dtype=pred_t.dtype)
    # Normalize axis weights to have mean of 1.0 to prevent scaling issues
    axis_weights = axis_weights / axis_weights.mean()
    translation_loss = (axis_weights * (pred_t - gt_t) ** 2).mean()
    translation_loss = torch.clamp(translation_loss, max=1e4)  # Prevent explosion
    trans_error = torch.norm(pred_t - gt_t, dim=-1)  # [B]

    # ── Rotation loss (smooth SO(3) surrogate: 1 - cos(theta)) ─────────────
    # R_diff = pred_R^T @ gt_R is identity when perfect.
    # trace(R_diff) = 1 + 2*cos(theta)  =>  cos(theta) = (trace - 1) / 2
    # Use 1 - cos(theta) instead of acos(theta) for smoother gradients,
    # especially near small angles while remaining geometry-aware on SO(3).
    R_diff = torch.bmm(pred_R.transpose(1, 2), gt_R)            # [B, 3, 3]
    trace = R_diff.diagonal(dim1=-2, dim2=-1).sum(dim=-1)      # [B]
    # Clamp more aggressively to prevent numerical issues
    cos_angle = torch.clamp((trace - 1.0) / 2.0, -0.9999, 0.9999)
    rotation_loss = (1.0 - cos_angle).mean()
    rotation_loss = torch.clamp(rotation_loss, max=1e4)
    # Safe acos for rotation error logging
    rotation_error_deg = torch.rad2deg(torch.acos(torch.clamp(cos_angle, -1.0, 1.0)))

    # ── BBox IoU loss ─────────────────────────────────────────────────────────
    pred_bbox = _transform_bbox_corners(bbox_corners, pred_transform)
    gt_bbox   = _transform_bbox_corners(bbox_corners, gt_transform)
    iou       = _aabb_iou(pred_bbox, gt_bbox)              # [B]
    bbox_corner_loss = (1.0 - iou).mean()
    bbox_corner_loss = torch.clamp(bbox_corner_loss, max=1e4)

    # ── Combine ────────────────────────────────────────────────────────────────
    total_loss = (
        trans_w * translation_loss
        + rot_w * rotation_loss
        + bbox_w * bbox_corner_loss
    )
    
    # Final NaN check before returning
    if torch.isnan(total_loss):
        # Fallback to prevent NaN propagation
        print("WARNING: NaN detected in pose_loss, using fallback")
        total_loss = torch.tensor(1.0, device=pred_t.device, dtype=pred_t.dtype)
    
    result = {
        "loss": total_loss,
        "translation": translation_loss.detach(),
        "rotation": rotation_loss.detach(),
        "bbox_corner": bbox_corner_loss.detach(),
        "rotation_error_deg": rotation_error_deg.mean().detach(),
    }
    
    # Optional confidence loss
    if pred_conf_t is not None and pred_conf_r is not None and conf_w > 0.0:
        conf_loss_dict = confidence_loss(
            pred_conf_t, pred_conf_r,
            trans_error, rotation_error_deg,
            temperature=conf_temp,
        )
        total_loss = total_loss + conf_w * conf_loss_dict["loss"]
        total_loss = torch.clamp(total_loss, max=1e4)
        result["loss"] = total_loss
        result["confidence"] = conf_loss_dict["loss"].detach()
        result["confidence_t"] = conf_loss_dict["conf_t"]
        result["confidence_r"] = conf_loss_dict["conf_r"]
    
    return result
