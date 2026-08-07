"""Shared fixtures. Importable without pytest - see tests/run_tests.py."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from PIL import Image

from preview_core.spec import Spec, default_spec_path

ROOT = Path(__file__).resolve().parent.parent
SPEC_PATH = default_spec_path()
ASSETS = ROOT / "assets"


def load_spec() -> Spec:
    return Spec.default()


def synth_render(
    w: int,
    h: int,
    canvas: int = 2000,
    shadow: Optional[Tuple[int, int]] = None,
    detail: bool = False,
) -> np.ndarray:
    """A rectangular 'model' on white, optionally with a soft floor shadow.

    Deliberately synthetic: geometry tests should fail because the solver is
    wrong, not because a photograph is ambiguous.
    """
    img = np.full((canvas, canvas, 3), 255, np.uint8)
    x0, y0 = (canvas - w) // 2, (canvas - h) // 2
    img[y0 : y0 + h, x0 : x0 + w] = (40, 50, 70)
    img[y0 + 12 : y0 + h - 12, x0 + 12 : x0 + w - 12] = (190, 170, 110)
    if detail:
        step = max(4, w // 40)
        img[y0 + 12 : y0 + h - 12 : step, x0 + 12 : x0 + w - 12] = (30, 30, 30)
    if shadow:
        sw, sh = shadow
        sx = x0 + (w - sw) // 2
        sy = y0 + h
        img[sy : sy + sh, sx : sx + sw] = (238, 238, 238)  # neutral, light, soft
    return img


def save(img: np.ndarray, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img).save(path)
    return path
