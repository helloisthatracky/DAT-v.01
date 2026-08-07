"""Step 12: export.

Metadata is stripped and the sRGB profile re-attached explicitly - a stray
Adobe RGB tag from the source render is a classic way to fail moderation
with a preview that looks perfect on the operator's screen.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image, ImageCms

from .spec import Spec

_SRGB_CACHE: bytes | None = None


def _srgb_profile() -> bytes:
    global _SRGB_CACHE
    if _SRGB_CACHE is None:
        _SRGB_CACHE = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    return _SRGB_CACHE


def export_jpeg(canvas: np.ndarray, path: str | Path, spec: Spec) -> Tuple[Path, int]:
    """Write a JPEG within the size budget. Returns (path, quality used)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    im = Image.fromarray(canvas, mode="RGB")

    kwargs = dict(
        format="JPEG",
        subsampling=spec.export.subsampling,
        optimize=True,
        progressive=False,
    )
    if spec.export.icc_srgb:
        kwargs["icc_profile"] = _srgb_profile()

    budget = spec.export.max_mb * 1e6
    q = spec.export.quality
    data = b""
    while True:
        buf = io.BytesIO()
        im.save(buf, quality=q, **kwargs)  # type: ignore[arg-type]
        data = buf.getvalue()
        if len(data) <= budget or q <= spec.export.min_quality:
            break
        q -= 2

    path.write_bytes(data)
    return path, q


def output_name(spec: Spec, model_id: str, index: int, role: str) -> str:
    return spec.export.name_template.format(model_id=model_id, index=index, role=role)
