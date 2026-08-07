"""Step 7: material spheres, main preview only.

Degradation ladder, in order: shrink the diameter to its floor, then drop
spheres one by one. Every downgrade produces a warning - the operator should
see that the layout was too tight, not silently get three spheres out of five.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

from .imaging import paste_alpha, resize_mask, resize_rgb, white_alpha
from .layout import union_rect
from .models import BBox, Placement, Transform
from .occupancy import FreeMap, scan_free_rects
from .spec import Spec

Orientation = str  # "v" | "h"


# --------------------------------------------------------------------------
# Candidate regions
# --------------------------------------------------------------------------
def _regions(
    spec: Spec, transforms: List[Transform], arrangement: str
) -> dict[str, Tuple[BBox, Orientation]]:
    gx0, gy0, gx1, gy1 = spec.guide_box()
    ux0, uy0, ux1, uy1 = union_rect(transforms)
    pad = spec.px(spec.spheres.clearance_ratio)

    out: dict[str, Tuple[BBox, Orientation]] = {
        "right_column": ((min(ux1 + pad, gx1), gy0, gx1, gy1), "v"),
        "left_column": ((gx0, gy0, max(ux0 - pad, gx0), gy1), "v"),
        "top_strip": ((gx0, gy0, gx1, max(uy0 - pad, gy0)), "h"),
        "bottom_strip": ((gx0, min(uy1 + pad, gy1), gx1, gy1), "h"),
    }
    if len(transforms) > 1:
        a, b = transforms[0].dst_solid, transforms[1].dst_solid
        if arrangement == "stacked":
            out["between_views"] = ((gx0, a[3], gx1, b[1]), "h")
        elif arrangement == "side_by_side":
            out["between_views"] = ((a[2], gy0, b[0], gy1), "v")
    return out


def _strip_size(n: int, d: int, gap: int, orient: Orientation) -> Tuple[int, int]:
    span = n * d + (n - 1) * gap
    return (d, span) if orient == "v" else (span, d)


def _score(
    box: BBox, region_name: str, region: BBox, orient: Orientation
) -> Tuple[float, float, int, int]:
    """Deterministic ranking.

    Primary: hug the outer edge of the region, so the strip reads as a
    deliberate column/row rather than something dropped in the middle.
    Secondary: centre along the strip's long axis - without this the run
    ends up jammed against whichever obstacle happens to come first.
    """
    x0, y0, x1, y1 = box
    if region_name == "right_column":
        primary = -x1
    elif region_name == "left_column":
        primary = float(x0)
    elif region_name == "top_strip":
        primary = float(y0)
    elif region_name == "bottom_strip":
        primary = -y1
    else:
        primary = float(y0)

    rx0, ry0, rx1, ry1 = region
    if orient == "v":
        centering = abs((y0 + y1) / 2 - (ry0 + ry1) / 2)
    else:
        centering = abs((x0 + x1) / 2 - (rx0 + rx1) / 2)
    return (float(primary), centering, y0, x0)


def find_strip(
    fm: FreeMap,
    spec: Spec,
    transforms: List[Transform],
    arrangement: str,
    model_type: str,
    n: int,
    d: int,
) -> Optional[Tuple[str, BBox, Orientation]]:
    gap = spec.px(spec.spheres.gap_ratio)
    step = max(2, spec.px(spec.spheres.search_step_ratio))
    regions = _regions(spec, transforms, arrangement)
    order: Sequence[str] = spec.spheres.priority.get(model_type, [])
    order = list(order) + [k for k in regions if k not in order] + ["anywhere"]

    for name in order:
        if name == "anywhere":
            candidates = [(spec.guide_box(), "v"), (spec.guide_box(), "h")]
        else:
            if name not in regions:
                continue
            candidates = [regions[name]]
        for region, orient in candidates:
            size = _strip_size(n, d, gap, orient)
            best = None
            for box in scan_free_rects(fm, region, size, step):
                key = _score(box, name, region, orient)
                if best is None or key < best[0]:
                    best = (key, box)
            if best is not None:
                return (name, best[1], orient)
    return None


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------
def _circle_alpha(d: int, ss: int = 4) -> np.ndarray:
    """Antialiased disc, built by supersampling then box-averaging."""
    big = d * ss
    yy, xx = np.mgrid[0:big, 0:big]
    r = (big - 1) / 2.0
    disc = ((xx - r) ** 2 + (yy - r) ** 2) <= r * r
    return disc.reshape(d, ss, d, ss).mean(axis=(1, 3)).astype(np.float32)


def render_sphere(
    canvas: np.ndarray, rgb: np.ndarray, alpha: Optional[np.ndarray], box: BBox, spec: Spec
) -> None:
    """Draw one material sphere, cropped to its own content first.

    Sphere renders usually arrive on a wide white plate. Resizing the plate
    straight to d x d turns every ball into an ellipse.
    """
    x0, y0, x1, y1 = box
    d = x1 - x0
    a_src = alpha if alpha is not None else white_alpha(rgb)
    ys, xs = np.nonzero(a_src > spec.mask.alpha_threshold)
    if xs.size:
        cx0, cy0, cx1, cy1 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
        rgb = rgb[cy0:cy1, cx0:cx1]
        alpha = alpha[cy0:cy1, cx0:cx1] if alpha is not None else None
    patch = resize_rgb(rgb, (d, d), spec.layout.resample)
    a = _circle_alpha(d)
    if alpha is not None:
        a = a * resize_mask(alpha, (d, d), spec.layout.resample)
    paste_alpha(canvas, patch, a, x0, y0)


def place_spheres(
    canvas: np.ndarray,
    fm: FreeMap,
    sphere_images: List[Tuple[np.ndarray, Optional[np.ndarray]]],
    spec: Spec,
    transforms: List[Transform],
    arrangement: str,
    model_type: str,
) -> Tuple[np.ndarray, List[Placement], List[str]]:
    warnings: List[str] = []
    if not spec.spheres.enabled or not sphere_images:
        return canvas, [], warnings

    n_target = min(len(sphere_images), spec.spheres.count)
    d0 = spec.px(spec.spheres.diameter_ratio)
    d_min = spec.px(spec.spheres.min_diameter_ratio)
    gap = spec.px(spec.spheres.gap_ratio)
    d_step = max(2, (d0 - d_min) // 6 or 2)

    found = None
    n = n_target
    while n >= 1 and found is None:
        d = d0
        while d >= d_min:
            hit = find_strip(fm, spec, transforms, arrangement, model_type, n, d)
            if hit:
                found = (n, d, hit)
                break
            d -= d_step
        if found is None:
            n -= 1

    if found is None:
        warnings.append("no free space for material spheres - none placed")
        return canvas, [], warnings

    n, d, (region_name, box, orient) = found
    if n < n_target:
        warnings.append(f"only {n} of {n_target} spheres fit - reduced")
    if d < d0:
        warnings.append(f"sphere diameter reduced {d0}px -> {d}px to fit")

    placements: List[Placement] = []
    x, y = box[0], box[1]
    for i in range(n):
        rect = (x, y, x + d, y + d)
        render_sphere(canvas, *sphere_images[i], rect, spec)
        placements.append(Placement(kind="sphere", index=i, rect=rect))
        if orient == "v":
            y += d + gap
        else:
            x += d + gap
    return canvas, placements, warnings
