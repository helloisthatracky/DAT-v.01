"""Step 11: the validator.

Deliberately usable on its own: the same checks run against a preview that
somebody made by hand in Photoshop. That is the moderation product hiding
inside the generator, and it costs almost nothing to expose.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from .geometry import _tight_bbox, shadow_mask, soft_mask
from .imaging import load_rgba
from .models import BBox, Check, Placement
from .spec import Spec

STRICT_WHITE_ROLES = ("secondary",)


def _bbox(binary: np.ndarray, spec: Spec) -> Optional[BBox]:
    """Same eps-filtered bbox the layout solver used.

    Must match geometry._tight_bbox exactly - measuring the result with a
    stricter rule than the one that placed it produces phantom overflows.
    """
    return _tight_bbox(binary, spec.mask.bbox_coverage_eps)


def _overlap(a: BBox, b: BBox) -> int:
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    return ix * iy


def validate(
    canvas: np.ndarray,
    role: str,
    spec: Spec,
    placements: Optional[List[Placement]] = None,
    file_path: Optional[str | Path] = None,
) -> List[Check]:
    """Run every guide rule against a finished canvas.

    Two rules are deliberately softer than the first draft of the tech spec.
    `tl_zone_clear` and `br_logo` fail only when *we* put something in a
    reserved corner; a model that legitimately spans the full guide box will
    always clip those corners, and the reference set confirms moderation
    accepts it. Model intrusion is reported as a warning instead.
    """
    placements = placements or []
    checks: List[Check] = []
    H, W = canvas.shape[:2]
    target = spec.canvas.size
    spec = spec.for_canvas(W, H)   # rules are ratios; re-base them on this image
    mask = soft_mask(canvas, None, spec)
    binary = mask > spec.mask.alpha_threshold
    zones = spec.zone_rects()

    # The model is everything we did not stamp on ourselves, minus the
    # contact shadow - which is allowed past the guides and would otherwise
    # register as an overflow the solver never made.
    model = binary & ~shadow_mask(canvas, mask, spec)
    for p in placements:
        x0, y0, x1, y1 = p.rect
        model[max(0, y0) : y1, max(0, x0) : x1] = False
    if not placements:
        bl = zones.get("BL")
        if bl:  # standalone validation: assume the brand logo owns BL
            model[bl[1] : bl[3], bl[0] : bl[2]] = False

    obj = _bbox(model, spec)
    gx0, gy0, gx1, gy1 = spec.guide_box()
    tol = spec.guides.tolerance(spec.canvas)
    is_closeup = role == "closeup"

    def add(cid: str, ok: bool, msg: str = "", level: str = "fail") -> None:
        checks.append(Check(cid, "pass" if ok else level, "" if ok else msg))

    def na(cid: str, why: str) -> None:
        checks.append(Check(cid, "pass", why))

    # 1. canvas_size ------------------------------------------------------
    # Exact match for our own output; a differently sized but correctly
    # proportioned preview is checked on its own terms and only warned about.
    if (W, H) == target:
        checks.append(Check("canvas_size", "pass"))
    elif abs(W / H - target[0] / target[1]) > 0.01:
        add("canvas_size", False, f"{W}x{H}: aspect differs from {target[0]}x{target[1]}")
    elif min(W, H) < spec.validate.min_canvas_side:
        add("canvas_size", False, f"{W}x{H} is below the {spec.validate.min_canvas_side}px floor")
    else:
        checks.append(
            Check("canvas_size", "warn", f"{W}x{H}, spec targets {target[0]}x{target[1]}")
        )

    # 2. bg_pure_white ----------------------------------------------------
    border = max(1, spec.px(spec.validate.bg_sample_border_ratio))
    ring = np.concatenate(
        [
            canvas[:border].reshape(-1, 3),
            canvas[-border:].reshape(-1, 3),
            canvas[:, :border].reshape(-1, 3),
            canvas[:, -border:].reshape(-1, 3),
        ]
    )
    dev = int(255 - ring.min())
    if is_closeup:
        na("bg_pure_white", "closeups bleed to the edge")
    else:
        strict = role == "secondary" or spec.validate.strict
        add(
            "bg_pure_white",
            dev <= spec.validate.bg_white_tolerance,
            f"background deviates by {dev}/255",
            "fail" if strict else "warn",
        )

    if obj is None:
        add("guides_touch", False, "empty canvas")
        return checks

    # 3-4. guides ---------------------------------------------------------
    if is_closeup:
        na("guides_touch", "n/a for closeups")
        na("guides_overflow", "n/a for closeups")
    else:
        touch = (
            abs(obj[0] - gx0) <= tol
            or abs(obj[2] - gx1) <= tol
            or abs(obj[1] - gy0) <= tol
            or abs(obj[3] - gy1) <= tol
        )
        add("guides_touch", touch, "object does not reach the guide frame")

        ov = spec.guides.max_overflow_px
        overflow = (
            obj[0] < gx0 - ov or obj[1] < gy0 - ov or obj[2] > gx1 + ov or obj[3] > gy1 + ov
        )
        add("guides_overflow", not overflow, f"object bbox {obj} crosses the guides")

    # 5. tl_zone_clear ----------------------------------------------------
    _zone_check(checks, "tl_zone_clear", "TL", zones, model, placements, is_closeup)

    # 6. br_logo ----------------------------------------------------------
    if spec.overlays.draw_3ddd_logo:
        add("br_logo", any(p.kind == "logo_3ddd" for p in placements), "3ddd logo missing")
    else:
        _zone_check(checks, "br_logo", "BR", zones, model, placements, is_closeup)

    # 7. no_spheres_on_2_3 ------------------------------------------------
    if role in ("secondary", "closeup"):
        add(
            "no_spheres_on_2_3",
            not any(p.kind == "sphere" for p in placements),
            "spheres belong on the main preview only",
        )
    else:
        na("no_spheres_on_2_3", "")

    # 8. sphere_overlap ---------------------------------------------------
    bad: List[str] = []
    for p in placements:
        if p.kind != "sphere":
            continue
        for zname, z in zones.items():
            if _overlap(p.rect, z) > 0:
                bad.append(f"sphere {p.index} overlaps zone {zname}")
    add("sphere_overlap", not bad, "; ".join(bad), "warn")

    # 9. single_instance --------------------------------------------------
    if role == "secondary":
        add("single_instance", _count_blobs(model) <= 1, "more than one object in frame")
    else:
        na("single_instance", "")

    # 10. no_frame --------------------------------------------------------
    if is_closeup:
        na("no_frame", "closeups bleed to the edge")
    else:
        add("no_frame", dev <= spec.validate.bg_white_tolerance, "dark border or vignette", "warn")

    # 11. file_limits -----------------------------------------------------
    if file_path is not None:
        p = Path(file_path)
        mb = p.stat().st_size / 1e6 if p.exists() else 0.0
        add(
            "file_limits",
            p.suffix.lower() in (".jpg", ".jpeg") and mb <= spec.export.max_mb,
            f"{p.suffix} {mb:.1f} MB exceeds {spec.export.max_mb} MB",
        )
    else:
        na("file_limits", "not exported yet")

    return checks


ZONE_MODEL_WARN = 0.60  # model may clip a reserved corner; covering it is a smell


def _zone_check(
    checks: List[Check],
    cid: str,
    zone: str,
    zones,
    model: np.ndarray,
    placements: List[Placement],
    is_closeup: bool,
) -> None:
    z = zones.get(zone)
    if z is None:
        return
    ours = [p for p in placements if p.kind != "brand_logo" and _overlap(p.rect, z) > 0]
    if zone == "BR":
        ours = [p for p in ours if p.kind != "logo_3ddd"]
    if ours:
        checks.append(
            Check(cid, "fail", f"{ours[0].kind} placed inside the reserved {zone} zone")
        )
        return
    if is_closeup:
        checks.append(Check(cid, "pass", "n/a for closeups"))
        return
    area = max(1, (z[2] - z[0]) * (z[3] - z[1]))
    cover = float(model[z[1] : z[3], z[0] : z[2]].sum()) / area
    if cover > ZONE_MODEL_WARN:
        checks.append(
            Check(cid, "warn", f"model covers {cover:.0%} of the {zone} zone - 3ddd will stamp over it")
        )
    else:
        checks.append(Check(cid, "pass"))


def _count_blobs(binary: np.ndarray, min_area_ratio: float = 0.002) -> int:
    """Coarse connected-component count on a downscaled mask.

    Downscaling is intentional - at full resolution a hanging pendant with a
    thin chain reads as several blobs, which is not what the rule is about.
    """
    from PIL import Image

    h, w = binary.shape
    small = np.asarray(
        Image.fromarray((binary * 255).astype(np.uint8)).resize(
            (max(1, w // 8), max(1, h // 8)), Image.Resampling.BILINEAR
        )
    ) > 40

    labels = np.zeros(small.shape, dtype=np.int32)
    cur = 0
    total = small.size
    for sy in range(small.shape[0]):
        for sx in range(small.shape[1]):
            if not small[sy, sx] or labels[sy, sx]:
                continue
            cur += 1
            stack = [(sy, sx)]
            labels[sy, sx] = cur
            count = 0
            while stack:
                y, x = stack.pop()
                count += 1
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < small.shape[0] and 0 <= nx < small.shape[1]:
                            if small[ny, nx] and not labels[ny, nx]:
                                labels[ny, nx] = cur
                                stack.append((ny, nx))
            if count / total < min_area_ratio:
                labels[labels == cur] = 0
                cur -= 1
    return cur


def validate_file(path: str | Path, role: str, spec: Spec) -> List[Check]:
    """Standalone entry point: check a preview nobody in this system made."""
    rgb, _ = load_rgba(path)
    return validate(rgb, role, spec, placements=[], file_path=path)
