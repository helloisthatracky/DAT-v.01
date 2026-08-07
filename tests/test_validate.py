"""Tests for step 11: the validator, including its standalone use.

The standalone path is the moderation product - it has to work on previews
this system never made.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import numpy as np

from preview_core.export import export_jpeg
from preview_core.models import Placement
from preview_core.validate import validate, validate_file
from tests.conftest import ASSETS, load_spec

SPEC = load_spec()
W, H = SPEC.canvas.width, SPEC.canvas.height


def _status(checks, cid: str) -> str:
    return next(c.status for c in checks if c.id == cid)


def _canvas_with_box(box) -> np.ndarray:
    c = np.full((H, W, 3), 255, np.uint8)
    x0, y0, x1, y1 = box
    c[y0:y1, x0:x1] = (40, 40, 40)
    return c


def _fitted_canvas() -> np.ndarray:
    gx0, gy0, gx1, gy1 = SPEC.guide_box()
    return _canvas_with_box((gx0, gy0 + 200, gx1, gy1 - 200))


def test_a_correctly_fitted_canvas_passes_everything():
    checks = validate(_fitted_canvas(), "main", SPEC)
    bad = [(c.id, c.status, c.msg) for c in checks if c.status != "pass"]
    assert not bad, bad


def test_wrong_canvas_size_fails():
    small = np.full((800, 800, 3), 255, np.uint8)
    small[200:600, 200:600] = 0
    assert _status(validate(small, "main", SPEC), "canvas_size") == "fail"


def test_object_short_of_the_guides_fails_guides_touch():
    gx0, gy0, gx1, gy1 = SPEC.guide_box()
    canvas = _canvas_with_box((gx0 + 200, gy0 + 200, gx1 - 200, gy1 - 200))
    assert _status(validate(canvas, "main", SPEC), "guides_touch") == "fail"


def test_object_past_the_guides_fails_guides_overflow():
    canvas = _canvas_with_box((5, 400, W - 5, H - 400))
    assert _status(validate(canvas, "main", SPEC), "guides_overflow") == "fail"


def test_grey_background_fails_on_secondary():
    canvas = _fitted_canvas()
    canvas[canvas[:, :, 0] == 255] = 250
    assert _status(validate(canvas, "secondary", SPEC), "bg_pure_white") == "fail"


def test_a_sphere_in_a_reserved_zone_fails():
    tl = SPEC.zone_rects()["TL"]
    placement = Placement(kind="sphere", index=0, rect=(tl[0] + 5, tl[1] + 5, tl[0] + 60, tl[1] + 60))
    checks = validate(_fitted_canvas(), "main", SPEC, placements=[placement])
    assert _status(checks, "tl_zone_clear") == "fail"
    assert _status(checks, "sphere_overlap") == "warn"


def test_model_clipping_a_corner_is_a_warning_not_a_failure():
    """A full-width model always grazes TL. Moderation accepts it; so do we."""
    gx0, gy0, gx1, gy1 = SPEC.guide_box()
    canvas = _canvas_with_box((gx0, gy0, gx1, gy1 - 300))
    assert _status(validate(canvas, "main", SPEC), "tl_zone_clear") == "warn"


def test_spheres_are_rejected_on_a_secondary():
    placement = Placement(kind="sphere", index=0, rect=(1200, 100, 1260, 160))
    checks = validate(_fitted_canvas(), "secondary", SPEC, placements=[placement])
    assert _status(checks, "no_spheres_on_2_3") == "fail"


def test_closeup_rules_are_relaxed():
    canvas = np.full((H, W, 3), 90, np.uint8)  # edge-to-edge content
    checks = validate(canvas, "closeup", SPEC)
    for cid in ("bg_pure_white", "guides_touch", "guides_overflow", "no_frame"):
        assert _status(checks, cid) == "pass", cid


def test_standalone_validation_of_a_file():
    tmp = Path(tempfile.mkdtemp(prefix="pk-val-"))
    try:
        path, _ = export_jpeg(_fitted_canvas(), tmp / "ready.jpg", SPEC)
        checks = validate_file(path, "main", SPEC)
        assert _status(checks, "canvas_size") == "pass"
        assert _status(checks, "file_limits") == "pass"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
