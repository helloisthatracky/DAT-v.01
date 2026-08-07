"""Step 5: paint the views onto a white canvas."""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from .imaging import blank_canvas, crop, paste_darken, resize_mask, resize_rgb
from .models import Transform, ViewAnalysis
from .spec import Spec


def compose(
    rgbs: List[np.ndarray],
    masks: List[np.ndarray],
    views: List[ViewAnalysis],
    transforms: List[Transform],
    spec: Spec,
) -> Tuple[np.ndarray, np.ndarray]:
    """Returns (canvas RGB uint8, canvas coverage mask float32).

    The coverage mask is what the occupancy stage needs; recomputing it from
    the finished canvas would also pick up the guides-free white background
    plus any JPEG noise, so we carry it forward instead.
    """
    canvas = blank_canvas(spec.canvas.width, spec.canvas.height, spec.canvas.background)
    cover = np.zeros((spec.canvas.height, spec.canvas.width), dtype=np.float32)

    by_index = {v.index: i for i, v in enumerate(views)}
    for t in transforms:
        i = by_index[t.view_index]
        v = views[i]
        patch = crop(rgbs[i], v.content_bbox)
        pmask = crop(masks[i], v.content_bbox)

        x0, y0, x1, y1 = t.dst_content
        size = (max(1, x1 - x0), max(1, y1 - y0))
        patch = resize_rgb(patch, size, spec.layout.resample)
        pmask = resize_mask(pmask, size, spec.layout.resample)

        paste_darken(canvas, patch, x0, y0)
        _max_into(cover, pmask, x0, y0)

    _force_background(canvas, cover, spec)
    return canvas, cover


def _max_into(dst: np.ndarray, patch: np.ndarray, x: int, y: int) -> None:
    dh, dw = dst.shape
    ph, pw = patch.shape
    sx0, sy0 = max(0, -x), max(0, -y)
    dx0, dy0 = max(0, x), max(0, y)
    dx1, dy1 = min(dw, x + pw), min(dh, y + ph)
    if dx1 <= dx0 or dy1 <= dy0:
        return
    sub = patch[sy0 : sy0 + (dy1 - dy0), sx0 : sx0 + (dx1 - dx0)]
    region = dst[dy0:dy1, dx0:dx1]
    np.maximum(region, sub, out=region)


def _force_background(canvas: np.ndarray, cover: np.ndarray, spec: Spec) -> None:
    """Pure #FFFFFF outside the object.

    Non-negotiable: 3ddd's reverse image search keys on a clean background,
    and JPEG noise around the silhouette measurably hurts it.
    """
    bg = cover <= 0.0
    canvas[bg] = np.array(spec.canvas.background, dtype=np.uint8)


def compose_closeup(
    rgb: np.ndarray, box: Tuple[int, int, int, int], spec: Spec
) -> np.ndarray:
    """Crop a fragment and blow it up to fill the canvas edge to edge.

    Guide frame is deliberately not applied here - closeups bleed.
    """
    x0, y0, x1, y1 = box
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(rgb.shape[1], x1), min(rgb.shape[0], y1)
    patch = rgb[y0:y1, x0:x1]

    # Square off the crop so the aspect ratio survives the resize.
    ph, pw = patch.shape[:2]
    side = max(ph, pw)
    square = blank_canvas(side, side, spec.canvas.background)
    paste_darken(square, patch, (side - pw) // 2, (side - ph) // 2)
    return resize_rgb(square, spec.canvas.size, spec.layout.resample)
