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


def _rotation_add_distance(
    pred_R: torch.Tensor,
    gt_R: torch.Tensor,
    points: torch.Tensor,
) -> torch.Tensor:
    """Average 3D distance (ADD) under pred vs GT rotation. Returns [B] in meters."""
    pred_pts = torch.bmm(points, pred_R.transpose(1, 2))
    gt_pts = torch.bmm(points, gt_R.transpose(1, 2))
    return torch.linalg.norm(pred_pts - gt_pts, dim=-1).mean(dim=-1)


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

    return inter_vol / (vol1 + vol2 - inter_vol + 1e-6)          # [B]


def confidence_loss(
    pred_conf_t: torch.Tensor,
    pred_conf_r: torch.Tensor,
    translation_error: torch.Tensor,
    rotation_error: torch.Tensor,
    temperature: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Train confidence heads as inverse of prediction error.
    
    Args:
        pred_conf_t: [B] predicted translation confidence in [0,1]
        pred_conf_r: [B] predicted rotation confidence in [0,1]
        translation_error: [B] translation L2 error in meters
        rotation_error: [B] rotation ADD error in meters
        temperature: scaling factor for exponential decay
    
    Returns:
        dict with 'loss', 'conf_t', 'conf_r' keys
    """
    # Target: confidence = exp(-temperature * error)
    target_conf_t = torch.exp(-temperature * translation_error.detach())
    target_conf_r = torch.exp(-temperature * rotation_error.detach())
    
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
    
    Uses axis-weighted translation loss, rotation ADD loss on bbox corners,
    and 3D IoU bbox loss, plus optional confidence loss.
    
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
    
    assert not torch.isnan(gt_transform).any(), "NaN in ground truth data!"
    assert not torch.isnan(bbox_corners).any(), "NaN in bbox corners data!"
    assert not torch.isnan(pred_transform).any(), "NaN in model predictions (exploding gradients?)"
    
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
    trans_sq_dist = ((pred_t - gt_t) ** 2).sum(dim=-1)
    trans_error = torch.sqrt(trans_sq_dist + 1e-6)  # [B]

    # ── Rotation loss (ADD on canonical bbox corners, rotation only) ─────────
    rotation_add_error = _rotation_add_distance(pred_R, gt_R, bbox_corners)  # [B]
    rotation_loss = rotation_add_error.mean()
    rotation_loss = torch.clamp(rotation_loss, max=1e4)
    rotation_error_deg = torch.rad2deg(rotation_geodesic_error(pred_R, gt_R))

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
            trans_error, rotation_add_error,
            temperature=conf_temp,
        )
        total_loss = total_loss + conf_w * conf_loss_dict["loss"]
        total_loss = torch.clamp(total_loss, max=1e4)
        result["loss"] = total_loss
        result["confidence"] = conf_loss_dict["loss"].detach()
        result["confidence_t"] = conf_loss_dict["conf_t"]
        result["confidence_r"] = conf_loss_dict["conf_r"]

        # Final NaN check before returning
    if torch.isnan(total_loss):
        # Fallback to prevent NaN propagation
        print("WARNING: NaN detected in pose_loss, using fallback")
        total_loss = (pred_transform*0).sum() + (pred_conf_t*0).sum() if pred_conf_t is not None else (pred_transform*0).sum()
        result["loss"] = total_loss
    
    return result


def scene_mask_bce_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """BCE-with-logits for per-scene-token masks. ``logits`` / ``target`` are [B, N]."""
    with torch.autocast(device_type=logits.device.type, enabled=False):
        return F.binary_cross_entropy_with_logits(
            logits.float(),
            target.to(device=logits.device, dtype=torch.float32),
            reduction="mean",
        )
