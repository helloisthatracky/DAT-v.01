"""Step 8: corner overlays.

Note on the 3ddd logo: `overlays.draw_3ddd_logo` defaults to false because
3ddd stamps its own watermark server-side (see overview.txt). We still keep
the BR zone clear and reserve BL for the brand logo. Flip the flag if the
moderation team confirms the logo must be baked in.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from .imaging import crop, paste_alpha, resize_mask, resize_rgb, white_alpha
from .models import BBox, Placement
from .spec import Spec, OverlaySlot

CORNERS = ("TL", "TR", "BL", "BR")


def _content_box(mask: np.ndarray, thr: float) -> BBox:
    ys, xs = np.nonzero(mask > thr)
    if xs.size == 0:
        return (0, 0, mask.shape[1], mask.shape[0])
    return (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)


def place_overlay(
    canvas: np.ndarray,
    rgb: np.ndarray,
    alpha: Optional[np.ndarray],
    slot: OverlaySlot,
    spec: Spec,
    kind: str,
) -> Tuple[np.ndarray, Optional[Placement]]:
    if not slot.enabled:
        return canvas, None

    a = alpha if alpha is not None else white_alpha(rgb)
    box = _content_box(a, spec.mask.alpha_threshold)
    patch = crop(rgb, box)
    pa = crop(a, box)

    tw = spec.px(slot.width_ratio)
    th = max(1, round(patch.shape[0] * tw / max(1, patch.shape[1])))
    patch = resize_rgb(patch, (tw, th), spec.layout.resample)
    pa = resize_mask(pa, (tw, th), spec.layout.resample)

    m = spec.px(slot.margin_ratio)
    W, H = spec.canvas.width, spec.canvas.height
    x = m if slot.corner in ("TL", "BL") else W - m - tw
    y = m if slot.corner in ("TL", "TR") else H - m - th

    paste_alpha(canvas, patch, pa, x, y)
    return canvas, Placement(kind=kind, index=0, rect=(x, y, x + tw, y + th))


def apply_overlays(
    canvas: np.ndarray,
    logo_3ddd: Optional[Tuple[np.ndarray, Optional[np.ndarray]]],
    brand_logo: Optional[Tuple[np.ndarray, Optional[np.ndarray]]],
    spec: Spec,
) -> Tuple[np.ndarray, List[Placement]]:
    out: List[Placement] = []
    if spec.overlays.draw_3ddd_logo and logo_3ddd is not None:
        canvas, p = place_overlay(
            canvas, logo_3ddd[0], logo_3ddd[1], spec.overlays.logo_3ddd, spec, "logo_3ddd"
        )
        if p:
            out.append(p)
    if brand_logo is not None and spec.overlays.brand_logo.enabled:
        canvas, p = place_overlay(
            canvas, brand_logo[0], brand_logo[1], spec.overlays.brand_logo, spec, "brand_logo"
        )
        if p:
            out.append(p)
    return canvas, out
