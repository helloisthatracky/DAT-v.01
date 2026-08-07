"""Step 10: propose interesting fragments for previews #4+.

Automatic selection is a *suggestion* only - per the risk register, the final
crop is always the art director's call in the UI. What the algorithm has to
get right is surfacing candidates that are on the model and detail-dense,
not empty background or flat panels.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from .imaging import luminance
from .models import BBox
from .spec import Spec

ENTROPY_BINS = 16


def _integral(a: np.ndarray) -> np.ndarray:
    out = np.zeros((a.shape[0] + 1, a.shape[1] + 1), dtype=np.float64)
    out[1:, 1:] = np.cumsum(np.cumsum(a.astype(np.float64), axis=0), axis=1)
    return out


def _box_sum(s: np.ndarray, x0: int, y0: int, x1: int, y1: int) -> float:
    return float(s[y1, x1] - s[y0, x1] - s[y1, x0] + s[y0, x0])


def _gradient_magnitude(gray: np.ndarray) -> np.ndarray:
    gy, gx = np.gradient(gray)
    return np.hypot(gx, gy).astype(np.float32)


def propose_closeups(
    rgb: np.ndarray, mask: np.ndarray, spec: Spec, n: int | None = None
) -> List[BBox]:
    n = n or spec.closeup.n
    h, w = mask.shape
    short = min(h, w)
    win = max(16, round(spec.closeup.window_ratio * short))
    step = max(4, round(spec.closeup.step_ratio * short))
    if win >= min(h, w):
        return []

    gray = luminance(rgb)
    grad = _gradient_magnitude(gray)
    solid = (mask > spec.mask.alpha_threshold).astype(np.float32)

    s_grad = _integral(grad)
    s_mask = _integral(solid)
    quant = np.clip((gray * (ENTROPY_BINS - 1)).round().astype(np.int32), 0, ENTROPY_BINS - 1)
    s_bins = [_integral((quant == b).astype(np.float32)) for b in range(ENTROPY_BINS)]

    area = float(win * win)
    cands: List[Tuple[float, BBox]] = []
    for y in range(0, h - win + 1, step):
        for x in range(0, w - win + 1, step):
            x1, y1 = x + win, y + win
            coverage = _box_sum(s_mask, x, y, x1, y1) / area
            if coverage < 0.5:  # mostly background - skip
                continue
            g = _box_sum(s_grad, x, y, x1, y1) / area
            counts = np.array([_box_sum(s, x, y, x1, y1) for s in s_bins]) / area
            counts = counts[counts > 0]
            ent = float(-(counts * np.log2(counts)).sum())
            cands.append((g, ent, (x, y, x1, y1)))  # type: ignore[arg-type]

    if not cands:
        return []

    gs = np.array([c[0] for c in cands], dtype=np.float64)
    es = np.array([c[1] for c in cands], dtype=np.float64)
    scores = (
        spec.closeup.gradient_weight * _norm(gs) + spec.closeup.entropy_weight * _norm(es)
    )

    order = sorted(
        range(len(cands)),
        key=lambda i: (-scores[i], cands[i][2][1], cands[i][2][0]),  # deterministic ties
    )
    picked: List[BBox] = []
    for i in order:
        box = cands[i][2]
        if all(_iou(box, p) < spec.closeup.nms_iou for p in picked):
            picked.append(box)
        if len(picked) >= n:
            break
    return picked


def _norm(a: np.ndarray) -> np.ndarray:
    lo, hi = float(a.min()), float(a.max())
    return (a - lo) / (hi - lo) if hi > lo else np.zeros_like(a)


def _iou(a: BBox, b: BBox) -> float:
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0, ix1 - ix0), max(0, iy1 - iy0)
    inter = iw * ih
    if inter == 0:
        return 0.0
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua
