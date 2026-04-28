"""Geometry helpers used by the v2 training loop."""

from __future__ import annotations

from typing import Any

import torch


def build_transform_from_Rt(rotation: torch.Tensor, translation: torch.Tensor) -> torch.Tensor:
    """Build a homogeneous transform from rotation and translation tensors."""
    batch_size = rotation.shape[0]
    transform = torch.eye(4, device=rotation.device, dtype=rotation.dtype).unsqueeze(0)
    transform = transform.expand(batch_size, -1, -1).clone()
    transform[:, :3, :3] = rotation
    transform[:, :3, 3] = translation
    return transform


def rotation_6d_to_matrix(rotation_6d: torch.Tensor) -> torch.Tensor:
    """Convert the 6D rotation representation to a 3x3 rotation matrix."""
    if rotation_6d.shape[-1] != 6:
        raise ValueError("rotation_6d must have shape [..., 6]")

    a1 = rotation_6d[..., 0:3]
    a2 = rotation_6d[..., 3:6]
    b1 = torch.nn.functional.normalize(a1, dim=-1)
    b2 = torch.nn.functional.normalize(a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)


def apply_transform(transform: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    """Apply a batch of homogeneous transforms to a batch of 3D points."""
    rotation = transform[:, :3, :3]
    translation = transform[:, :3, 3].unsqueeze(1)
    return torch.matmul(points, rotation.transpose(-1, -2)) + translation


def coerce_pose_output(model_output: Any) -> torch.Tensor:
    """Normalize common model output formats into a [B, 4, 4] transform."""
    if isinstance(model_output, torch.Tensor):
        if model_output.ndim == 3 and model_output.shape[-2:] == (4, 4):
            return model_output
        raise ValueError("Tensor model outputs must already be [B, 4, 4]")

    if isinstance(model_output, dict):
        if "transform" in model_output:
            transform = model_output["transform"]
            if not isinstance(transform, torch.Tensor):
                raise TypeError("model_output['transform'] must be a tensor")
            return transform
        if "pred_transform" in model_output:
            transform = model_output["pred_transform"]
            if not isinstance(transform, torch.Tensor):
                raise TypeError("model_output['pred_transform'] must be a tensor")
            return transform
        if {"rotation", "translation"}.issubset(model_output):
            return build_transform_from_Rt(model_output["rotation"], model_output["translation"])
        if {"pred_R", "pred_t"}.issubset(model_output):
            return build_transform_from_Rt(model_output["pred_R"], model_output["pred_t"])
        if {"rotation_6d", "translation"}.issubset(model_output):
            return build_transform_from_Rt(
                rotation_6d_to_matrix(model_output["rotation_6d"]),
                model_output["translation"],
            )

    if isinstance(model_output, (tuple, list)):
        if len(model_output) == 2:
            first, second = model_output
            if isinstance(first, torch.Tensor) and isinstance(second, torch.Tensor):
                if first.ndim >= 2 and first.shape[-2:] == (3, 3):
                    return build_transform_from_Rt(first, second)
                if second.ndim >= 2 and second.shape[-2:] == (3, 3):
                    return build_transform_from_Rt(second, first)
        if len(model_output) == 3:
            rotation, translation, _ = model_output
            if isinstance(rotation, torch.Tensor) and isinstance(translation, torch.Tensor):
                return build_transform_from_Rt(rotation, translation)

    raise TypeError(f"Unsupported model output type: {type(model_output)!r}")


def rotation_geodesic_error(pred_rotation: torch.Tensor, gt_rotation: torch.Tensor) -> torch.Tensor:
    """Return the per-sample geodesic rotation error in radians."""
    relative = pred_rotation @ gt_rotation.transpose(-1, -2)
    trace = relative.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    cosine = torch.clamp((trace - 1.0) * 0.5, -1.0 + 1e-7, 1.0 - 1e-7)
    return torch.acos(cosine)