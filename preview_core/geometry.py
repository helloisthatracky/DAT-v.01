"""Steps 1-2 of the pipeline: soft mask, shadow-aware bbox, model type."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .imaging import luminance, median_filter, saturation, white_alpha
from .models import BBox, ViewAnalysis
from .spec import Spec


def soft_mask(rgb: np.ndarray, alpha: Optional[np.ndarray], spec: Spec) -> np.ndarray:
    """Step 1. Alpha if present, white-background estimate otherwise."""
    m = alpha.astype(np.float32) if alpha is not None else white_alpha(rgb)
    m = median_filter(m, spec.mask.median_size)
    m[m < spec.mask.alpha_threshold] = 0.0
    return m


def shadow_mask(rgb: np.ndarray, mask: np.ndarray, spec: Spec) -> np.ndarray:
    """Contact shadow: near-neutral, light, soft, and below the centre of mass.

    Excluded from the fit bbox, kept in the frame. Without this a soft floor
    shadow silently inflates the bbox and the model stops touching the guides.
    """
    if not spec.shadow.enabled:
        return np.zeros_like(mask, dtype=bool)

    solid = mask > spec.mask.alpha_threshold
    if not solid.any():
        return np.zeros_like(mask, dtype=bool)

    cand = (
        solid
        & (saturation(rgb) < spec.shadow.max_saturation)
        & (luminance(rgb) > spec.shadow.min_luminance)
        & (mask < spec.shadow.max_alpha)
    )
    if spec.shadow.below_centroid:
        ys = np.nonzero(solid)[0]
        cy = float(ys.mean())
        rows = np.arange(mask.shape[0])[:, None]
        cand &= rows > cy
    return cand


def _tight_bbox(binary: np.ndarray, eps: float) -> Optional[BBox]:
    """Bbox that ignores rows/columns whose coverage is below `eps`.

    Guards against JPEG ringing and stray single pixels that would otherwise
    move the bbox by tens of pixels.
    """
    if not binary.any():
        return None
    h, w = binary.shape
    col = binary.sum(axis=0) / max(h, 1)
    row = binary.sum(axis=1) / max(w, 1)
    xs = np.nonzero(col > eps)[0]
    ys = np.nonzero(row > eps)[0]
    if xs.size == 0 or ys.size == 0:  # everything below eps - fall back to raw
        xs = np.nonzero(col > 0)[0]
        ys = np.nonzero(row > 0)[0]
        if xs.size == 0 or ys.size == 0:
            return None
    return (int(xs[0]), int(ys[0]), int(xs[-1]) + 1, int(ys[-1]) + 1)


def analyse_view(
    index: int, path: str, rgb: np.ndarray, alpha: Optional[np.ndarray], spec: Spec
) -> Tuple[ViewAnalysis, np.ndarray]:
    """Step 2. Returns the analysis plus the soft mask (reused downstream)."""
    mask = soft_mask(rgb, alpha, spec)
    binary = mask > spec.mask.alpha_threshold
    content = _tight_bbox(binary, spec.mask.bbox_coverage_eps)
    if content is None:
        raise ValueError(f"view {index}: empty frame, no object found")

    shadow = shadow_mask(rgb, mask, spec)
    solid = _tight_bbox(binary & ~shadow, spec.mask.bbox_coverage_eps) or content

    h, w = mask.shape
    return (
        ViewAnalysis(
            index=index,
            path=path,
            size=(w, h),
            solid_bbox=solid,
            content_bbox=content,
            shadow_ratio=float(shadow.sum()) / max(float(binary.sum()), 1.0),
            fill_ratio=float(binary.mean()),
        ),
        mask,
    )


def classify(ratio: float, spec: Spec) -> str:
    """V = vertical, H = horizontal, S = near-square."""
    if ratio >= spec.classify.h_threshold:
        return "H"
    if ratio <= spec.classify.v_threshold:
        return "V"
    return "S"
