"""Pose losses for the v2 training engine."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .config import LossConfig
from .geometry import apply_transform, rotation_geodesic_error


@dataclass(frozen=True)
class PoseLossWeights:
    translation_weight: float = 1.0
    rotation_weight: float = 0.5
    bbox_corner_weight: float = 2.0


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
    
    Args:
        pred_transform: [B, 4, 4] predicted transformation matrices
        gt_transform: [B, 4, 4] ground truth transformation matrices
        bbox_corners: [B, K, 3] object bbox corners in model frame
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
    
    # Translation loss
    pred_translation = pred_transform[:, :3, 3]
    gt_translation = gt_transform[:, :3, 3]
    trans_error = torch.norm(pred_translation - gt_translation, dim=-1)  # [B]
    translation_loss = F.smooth_l1_loss(pred_translation, gt_translation)

    # Rotation loss
    pred_rotation = pred_transform[:, :3, :3]
    gt_rotation = gt_transform[:, :3, :3]
    rotation_error = rotation_geodesic_error(pred_rotation, gt_rotation)  # [B]
    rotation_error_deg = torch.rad2deg(rotation_error)
    rotation_loss = rotation_error.mean()

    # Bbox corner loss
    pred_corners = apply_transform(pred_transform, bbox_corners)
    gt_corners = apply_transform(gt_transform, bbox_corners)
    bbox_corner_loss = F.smooth_l1_loss(pred_corners, gt_corners)

    # Total loss
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
        "translation_error_m": trans_error.mean().detach(),
    }
    
    # Optional confidence loss
    if pred_conf_t is not None and pred_conf_r is not None and conf_w > 0.0:
        conf_loss_dict = confidence_loss(
            pred_conf_t, pred_conf_r,
            trans_error, rotation_error_deg,
            temperature=conf_temp,
        )
        total_loss = total_loss + conf_w * conf_loss_dict["loss"]
        result["loss"] = total_loss
        result["confidence"] = conf_loss_dict["loss"].detach()
        result["confidence_t"] = conf_loss_dict["conf_t"]
        result["confidence_r"] = conf_loss_dict["conf_r"]
    
    return result
