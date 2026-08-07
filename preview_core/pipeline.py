"""The whole tract, steps 0-12, in one deterministic pass.

Same input -> byte-identical output. No seeds, no samplers, no network.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from .closeup import propose_closeups
from .compose import compose, compose_closeup
from .export import export_jpeg, output_name
from .geometry import analyse_view, classify
from .imaging import InputError, load_rgba
from .layout import Preset, pick_preset, solve_layout, union_rect
from .models import Job, Report, ResultItem, Transform, ViewAnalysis
from .occupancy import build_free_map
from .overlay import apply_overlays
from .spec import Spec
from .spheres import place_spheres
from .validate import validate

Loaded = Tuple[np.ndarray, Optional[np.ndarray]]


def _load_optional(path: Optional[str]) -> Optional[Loaded]:
    if not path:
        return None
    try:
        return load_rgba(path)
    except (InputError, FileNotFoundError):
        return None


def build_previews(
    job: Job,
    spec: Spec,
    out_dir: str | Path,
    logo_3ddd_path: Optional[str] = None,
) -> Report:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.perf_counter()

    report = Report(
        job_id=job.job_id,
        model_id=job.model_id,
        spec_version=spec.version,
        spec_digest=spec.digest,
    )

    # -- Step 0-2: intake and analysis ------------------------------------
    rgbs: List[np.ndarray] = []
    masks: List[np.ndarray] = []
    views: List[ViewAnalysis] = []
    for i, v in enumerate(job.views):
        try:
            rgb, alpha = load_rgba(v.url)
        except (InputError, FileNotFoundError) as exc:
            report.warnings.append(f"view {i} skipped: {exc}")
            continue
        analysis, mask = analyse_view(len(views), v.url, rgb, alpha, spec)
        if min(rgb.shape[:2]) < spec.canvas.short_side:
            report.warnings.append(
                f"view {i}: {rgb.shape[1]}x{rgb.shape[0]} is below the "
                f"{spec.canvas.short_side}px canvas - upscaling is unavoidable"
            )
        rgbs.append(rgb)
        masks.append(mask)
        views.append(analysis)

    if not views:
        report.warnings.append("no usable views - nothing produced")
        return report

    model_type = classify(views[0].ratio, spec)
    report.detected = {
        "type": model_type,
        "ratio": round(views[0].ratio, 3),
        "bbox": list(views[0].solid_bbox),
        "shadow_ratio": round(views[0].shadow_ratio, 4),
        "views": len(views),
    }

    spheres = [s for s in (_load_optional(p) for p in job.spheres) if s]
    brand_logo = _load_optional(job.brand_logo)
    logo_3ddd = _load_optional(logo_3ddd_path)

    index = 1

    # -- Steps 3-8: main preview ------------------------------------------
    if "main" in job.outputs:
        n_views = min(len(views), int(job.overrides.get("max_views", 2)))
        preset = pick_preset(model_type, n_views, "main")
        transforms, preset, warns = solve_layout(views[:n_views], preset, spec)
        canvas, cover = compose(rgbs, masks, views, transforms, spec)

        fm = build_free_map(cover, spec)
        n_spheres = int(job.overrides.get("sphere_count", spec.spheres.count))
        canvas, placements, sw = place_spheres(
            canvas, fm, spheres[:n_spheres], spec, transforms, preset.arrangement, model_type
        )
        canvas, ov = apply_overlays(canvas, logo_3ddd, brand_logo, spec)
        placements += ov

        path, _ = export_jpeg(canvas, out_dir / output_name(spec, job.model_id, index, "main"), spec)
        report.results.append(
            ResultItem(
                role="main",
                index=index,
                path=str(path),
                checks=validate(canvas, "main", spec, placements, path),
                placements=placements,
                warnings=warns + sw,
            )
        )
        index += 1

        # Closeups are only sensible when the model sits well inside the frame.
        free_open = build_free_map(cover, spec, include_zones=False).free_ratio(spec.guide_box())
        closeups_ok = free_open >= spec.closeup.min_free_area_ratio
    else:
        closeups_ok = True

    # -- Step 9: secondary previews ---------------------------------------
    if "secondary" in job.outputs:
        for v in views[:2]:
            preset = pick_preset(model_type, 1, "secondary")
            transforms, preset, warns = solve_layout([v], preset, spec)
            canvas, _ = compose(rgbs, masks, views, transforms, spec)
            canvas, ov = apply_overlays(canvas, logo_3ddd, None, spec)
            path, _ = export_jpeg(
                canvas, out_dir / output_name(spec, job.model_id, index, "secondary"), spec
            )
            report.results.append(
                ResultItem(
                    role="secondary",
                    index=index,
                    path=str(path),
                    checks=validate(canvas, "secondary", spec, ov, path),
                    placements=ov,
                    warnings=warns,
                )
            )
            index += 1

    # -- Step 10: closeups -------------------------------------------------
    if "closeup" in job.outputs:
        if not closeups_ok:
            report.warnings.append(
                "closeups skipped: the main layout leaves too little free space"
            )
        else:
            boxes = propose_closeups(rgbs[0], masks[0], spec)
            if not boxes:
                report.warnings.append("closeups skipped: no detail-dense fragment found")
            for box in boxes:
                canvas = compose_closeup(rgbs[0], box, spec)
                canvas, ov = apply_overlays(canvas, logo_3ddd, None, spec)
                path, _ = export_jpeg(
                    canvas, out_dir / output_name(spec, job.model_id, index, "closeup"), spec
                )
                report.results.append(
                    ResultItem(
                        role="closeup",
                        index=index,
                        path=str(path),
                        checks=validate(canvas, "closeup", spec, ov, path),
                        placements=ov,
                        warnings=[f"source crop {box} - confirm manually"],
                    )
                )
                index += 1

    report.detected["elapsed_s"] = round(time.perf_counter() - t_start, 3)
    return report
