"""Steps 3-4: preset choice and the layout solver.

The solver's single hard rule, taken straight from the guide: the composed
content touches the guide frame on exactly one axis - the limiting one. Both
views always share one scale factor, so two angles of the same model stay
comparable in size.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

from .models import BBox, Transform, ViewAnalysis
from .spec import Spec

ARRANGEMENTS = ("single", "side_by_side", "stacked")


@dataclass(frozen=True)
class Preset:
    name: str
    arrangement: str
    model_type: str
    role: str


def pick_preset(model_type: str, n_views: int, role: str) -> Preset:
    """Guide table 3.1: (type, views) -> arrangement.

    V pairs read best side by side, H pairs stacked. S falls back to the
    arrangement that yields the larger scale - decided later by the solver.
    """
    if role != "main" or n_views <= 1:
        return Preset(f"{model_type}_1_{role}", "single", model_type, role)
    table = {"V": "side_by_side", "H": "stacked", "S": "auto"}
    return Preset(f"{model_type}_2_{role}", table[model_type], model_type, role)


def _scaled(box: BBox, s: float) -> Tuple[int, int]:
    x0, y0, x1, y1 = box
    return (max(1, round((x1 - x0) * s)), max(1, round((y1 - y0) * s)))


def _fit_scale(uw: float, uh: float, gw: float, gh: float, spec: Spec) -> float:
    return min(gw / uw, gh / uh, spec.layout.max_upscale)


def _union_scale(views: List[ViewAnalysis], arrangement: str, spec: Spec) -> Tuple[float, float, float]:
    """Common scale plus the resulting union size, in canvas pixels."""
    gx0, gy0, gx1, gy1 = spec.guide_box()
    gw, gh = gx1 - gx0, gy1 - gy0
    gap = spec.px(spec.layout.gap_ratio)
    dims = [v.solid_wh for v in views]

    if arrangement == "single" or len(views) == 1:
        w, h = dims[0]
        s = _fit_scale(w, h, gw, gh, spec)
        return s, w * s, h * s
    if arrangement == "side_by_side":
        sw = sum(d[0] for d in dims)
        mh = max(d[1] for d in dims)
        s = _fit_scale(sw, mh, gw - gap * (len(views) - 1), gh, spec)
        return s, sw * s + gap * (len(views) - 1), mh * s
    # stacked
    mw = max(d[0] for d in dims)
    sh = sum(d[1] for d in dims)
    s = _fit_scale(mw, sh, gw, gh - gap * (len(views) - 1), spec)
    return s, mw * s, sh * s + gap * (len(views) - 1)


def solve_layout(
    views: List[ViewAnalysis], preset: Preset, spec: Spec
) -> Tuple[List[Transform], Preset, List[str]]:
    """Returns transforms, the preset actually used, and any warnings."""
    warnings: List[str] = []
    gx0, gy0, gx1, gy1 = spec.guide_box()
    gw, gh = gx1 - gx0, gy1 - gy0
    gap = spec.px(spec.layout.gap_ratio)

    arrangement = preset.arrangement
    if arrangement == "auto":
        cands = [(a, _union_scale(views, a, spec)) for a in ("side_by_side", "stacked")]
        arrangement = max(cands, key=lambda c: c[1][0])[0]
        preset = Preset(preset.name, arrangement, preset.model_type, preset.role)

    scale, uw, uh = _union_scale(views, arrangement, spec)

    # Two views that end up too small are worse than one honest view.
    if len(views) > 1 and (uw * uh) / (gw * gh) < spec.layout.two_views_min_fill:
        warnings.append(
            "two views fill only "
            f"{(uw * uh) / (gw * gh):.0%} of the guide box - falling back to one view"
        )
        views = views[:1]
        preset = Preset(f"{preset.model_type}_1_{preset.role}", "single", preset.model_type, preset.role)
        arrangement = "single"
        scale, uw, uh = _union_scale(views, arrangement, spec)

    if scale >= spec.layout.max_upscale - 1e-9:
        warnings.append(
            f"render too small: capped at {spec.layout.max_upscale:g}x upscale, "
            "guides are not touched"
        )

    # Union origin inside the guide box.
    ox = gx0 + (gw - uw) / 2.0
    oy = gy0 + (gh - uh) * spec.layout.y_anchor

    transforms: List[Transform] = []
    cx, cy = ox, oy
    for v in views:
        sw, sh = _scaled(v.solid_bbox, scale)
        if arrangement == "side_by_side":
            x, y = cx, oy + (uh - sh) / 2.0
            cx += sw + gap
        elif arrangement == "stacked":
            x, y = ox + (uw - sw) / 2.0, cy
            cy += sh + gap
        else:
            x, y = ox, oy

        dst_solid = (round(x), round(y), round(x) + sw, round(y) + sh)
        transforms.append(
            Transform(
                view_index=v.index,
                scale=scale,
                dst_solid=dst_solid,
                dst_content=_content_rect(v, dst_solid, scale),
            )
        )
    return transforms, preset, warnings


def _content_rect(v: ViewAnalysis, dst_solid: BBox, s: float) -> BBox:
    """Where the full crop (shadow included) lands, anchored on the solid bbox."""
    sx0, sy0, sx1, sy1 = v.solid_bbox
    cx0, cy0, cx1, cy1 = v.content_bbox
    dx0, dy0 = dst_solid[0], dst_solid[1]
    x0 = dx0 - round((sx0 - cx0) * s)
    y0 = dy0 - round((sy0 - cy0) * s)
    return (x0, y0, x0 + max(1, round((cx1 - cx0) * s)), y0 + max(1, round((cy1 - cy0) * s)))


def union_rect(transforms: List[Transform], attr: str = "dst_solid") -> BBox:
    boxes = [getattr(t, attr) for t in transforms]
    return (
        min(b[0] for b in boxes),
        min(b[1] for b in boxes),
        max(b[2] for b in boxes),
        max(b[3] for b in boxes),
    )
