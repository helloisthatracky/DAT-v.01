"""Unit tests for step 1-2: mask, bbox, shadow, classification."""

from __future__ import annotations

import numpy as np

from preview_core.geometry import analyse_view, classify, shadow_mask, soft_mask
from tests.conftest import load_spec, synth_render

SPEC = load_spec()


def test_classify_thresholds():
    assert classify(2.0, SPEC) == "H"
    assert classify(SPEC.classify.h_threshold, SPEC) == "H"
    assert classify(0.4, SPEC) == "V"
    assert classify(SPEC.classify.v_threshold, SPEC) == "V"
    assert classify(1.0, SPEC) == "S"


def test_bbox_is_exact_on_a_clean_render():
    img = synth_render(600, 400, canvas=1200)
    view, _ = analyse_view(0, "synthetic", img, None, SPEC)
    assert view.solid_bbox == (300, 400, 900, 800)
    assert view.solid_wh == (600, 400)
    assert abs(view.ratio - 1.5) < 1e-6


def test_alpha_channel_takes_priority_over_the_white_estimate():
    img = synth_render(400, 400, canvas=800)
    alpha = np.zeros((800, 800), np.float32)
    alpha[100:300, 100:300] = 1.0  # deliberately disagrees with the pixels
    view, _ = analyse_view(0, "synthetic", img, alpha, SPEC)
    assert view.solid_bbox == (100, 100, 300, 300)


def test_shadow_is_excluded_from_the_fit_bbox():
    img = synth_render(600, 400, canvas=1400, shadow=(500, 120))
    view, mask = analyse_view(0, "synthetic", img, None, SPEC)
    sx0, sy0, sx1, sy1 = view.solid_bbox
    cx0, cy0, cx1, cy1 = view.content_bbox

    assert (sx1 - sx0, sy1 - sy0) == (600, 400)   # shadow did not inflate the fit
    assert cy1 > sy1                               # but it is still in the frame
    assert view.shadow_ratio > 0.05


def test_shadow_detector_ignores_a_saturated_object():
    img = np.full((600, 600, 3), 255, np.uint8)
    img[200:400, 200:400] = (240, 120, 40)  # light but strongly saturated
    mask = soft_mask(img, None, SPEC)
    assert shadow_mask(img, mask, SPEC).sum() == 0


def test_speckle_noise_does_not_move_the_bbox():
    img = synth_render(600, 400, canvas=1200)
    rng = np.random.default_rng(0)
    ys = rng.integers(0, 1200, 40)
    xs = rng.integers(0, 1200, 40)
    img[ys, xs] = 250  # isolated near-white pixels, JPEG-ringing style
    view, _ = analyse_view(0, "synthetic", img, None, SPEC)
    assert view.solid_bbox == (300, 400, 900, 800)
