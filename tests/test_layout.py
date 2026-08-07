"""Unit tests for steps 3-4: preset choice and the fit solver."""

from __future__ import annotations

from preview_core.geometry import analyse_view
from preview_core.layout import pick_preset, solve_layout, union_rect
from tests.conftest import load_spec, synth_render

SPEC = load_spec()
TOL = SPEC.guides.tolerance(SPEC.canvas)


def _view(w: int, h: int, canvas: int = 2400, idx: int = 0):
    img = synth_render(w, h, canvas=canvas)
    return analyse_view(idx, f"v{idx}", img, None, SPEC)[0]


def _touches(box, spec=SPEC) -> bool:
    gx0, gy0, gx1, gy1 = spec.guide_box()
    return (
        abs(box[0] - gx0) <= TOL
        or abs(box[2] - gx1) <= TOL
        or abs(box[1] - gy0) <= TOL
        or abs(box[3] - gy1) <= TOL
    )


def _inside(box, spec=SPEC) -> bool:
    gx0, gy0, gx1, gy1 = spec.guide_box()
    ov = spec.guides.max_overflow_px
    return box[0] >= gx0 - ov and box[1] >= gy0 - ov and box[2] <= gx1 + ov and box[3] <= gy1 + ov


def test_preset_table():
    assert pick_preset("V", 2, "main").arrangement == "side_by_side"
    assert pick_preset("H", 2, "main").arrangement == "stacked"
    assert pick_preset("S", 2, "main").arrangement == "auto"
    assert pick_preset("V", 2, "secondary").arrangement == "single"
    assert pick_preset("H", 1, "main").arrangement == "single"


def test_single_view_touches_exactly_the_limiting_axis():
    v = _view(1600, 900)
    tf, _, warns = solve_layout([v], pick_preset("H", 1, "main"), SPEC)
    box = union_rect(tf)
    gx0, gy0, gx1, gy1 = SPEC.guide_box()
    assert not warns
    assert abs(box[0] - gx0) <= TOL and abs(box[2] - gx1) <= TOL   # width limits
    assert box[1] > gy0 + TOL and box[3] < gy1 - TOL               # height has slack
    assert _inside(box)


def test_vertical_model_touches_the_vertical_axis():
    v = _view(700, 1800)
    tf, _, _ = solve_layout([v], pick_preset("V", 1, "main"), SPEC)
    box = union_rect(tf)
    gx0, gy0, gx1, gy1 = SPEC.guide_box()
    assert abs(box[1] - gy0) <= TOL and abs(box[3] - gy1) <= TOL
    assert _inside(box)


def test_two_views_share_one_scale():
    a, b = _view(800, 1700, idx=0), _view(600, 1700, idx=1)
    tf, preset, _ = solve_layout([a, b], pick_preset("V", 2, "main"), SPEC)
    assert preset.arrangement == "side_by_side"
    assert abs(tf[0].scale - tf[1].scale) < 1e-9
    assert tf[0].dst_solid[2] <= tf[1].dst_solid[0]     # no horizontal overlap
    assert _inside(union_rect(tf))


def test_stacked_views_do_not_overlap():
    a, b = _view(1800, 600, idx=0), _view(1800, 700, idx=1)
    tf, preset, _ = solve_layout([a, b], pick_preset("H", 2, "main"), SPEC)
    assert preset.arrangement == "stacked"
    assert tf[0].dst_solid[3] <= tf[1].dst_solid[1]
    assert _inside(union_rect(tf))


def test_auto_arrangement_picks_the_larger_scale():
    a, b = _view(1400, 1400, idx=0), _view(1400, 1400, idx=1)
    tf, preset, _ = solve_layout([a, b], pick_preset("S", 2, "main"), SPEC)
    assert preset.arrangement in ("side_by_side", "stacked")
    assert _inside(union_rect(tf))


def test_two_views_that_would_be_tiny_fall_back_to_one():
    a, b = _view(2000, 260, idx=0), _view(2000, 260, idx=1)
    tf, preset, warns = solve_layout([a, b], pick_preset("H", 2, "main"), SPEC)
    assert preset.arrangement == "single"
    assert len(tf) == 1
    assert any("falling back" in w for w in warns)


def test_upscale_is_capped_and_reported():
    v = _view(200, 200, canvas=600)
    tf, _, warns = solve_layout([v], pick_preset("S", 1, "main"), SPEC)
    assert tf[0].scale <= SPEC.layout.max_upscale + 1e-9
    assert any("upscale" in w for w in warns)
    assert not _touches(union_rect(tf))  # honest: it cannot reach the guides
