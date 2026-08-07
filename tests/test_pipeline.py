"""End-to-end tests for the full tract, steps 0-12."""

from __future__ import annotations

import hashlib
import shutil
import tempfile
from pathlib import Path

from preview_core.models import Job, View
from preview_core.pipeline import build_previews
from tests.conftest import ASSETS, load_spec, save, synth_render

SPEC = load_spec()
MODEL = ASSETS / "model.png"
SPHERE = ASSETS / "watermark.png"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tmp() -> Path:
    return Path(tempfile.mkdtemp(prefix="pk-"))


def test_real_model_produces_a_clean_set():
    out = _tmp()
    try:
        job = Job(
            job_id="t1",
            model_id="12345",
            views=[View(url=str(MODEL), angle="three_quarter")],
            spheres=[str(SPHERE)] * 4,
            outputs=["main", "secondary", "closeup"],
        )
        report = build_previews(job, SPEC, out)

        assert report.detected["type"] == "H"
        assert report.status != "failed"
        assert [r.role for r in report.results][:2] == ["main", "secondary"]

        failures = [
            (r.role, c.id, c.msg)
            for r in report.results
            for c in r.checks
            if c.status == "fail"
        ]
        assert not failures, failures

        for r in report.results:
            assert Path(r.path).exists()
            assert Path(r.path).stat().st_size > 0
    finally:
        shutil.rmtree(out, ignore_errors=True)


def test_output_is_byte_identical_between_runs():
    """The determinism requirement, checked the only way that counts."""
    a, b = _tmp(), _tmp()
    try:
        job = Job(
            job_id="t2",
            model_id="12345",
            views=[View(url=str(MODEL))],
            spheres=[str(SPHERE)] * 3,
            outputs=["main", "secondary", "closeup"],
        )
        ra = build_previews(job, SPEC, a)
        rb = build_previews(job, SPEC, b)

        assert [r.path for r in ra.results] != []
        for x, y in zip(ra.results, rb.results):
            assert _sha(Path(x.path)) == _sha(Path(y.path)), x.role
    finally:
        shutil.rmtree(a, ignore_errors=True)
        shutil.rmtree(b, ignore_errors=True)


def test_file_names_follow_the_template():
    out = _tmp()
    try:
        job = Job(
            job_id="t3",
            model_id="777",
            views=[View(url=str(MODEL))],
            outputs=["main", "secondary"],
        )
        report = build_previews(job, SPEC, out)
        names = [Path(r.path).name for r in report.results]
        assert names[0] == "777_01_main.jpg"
        assert names[1] == "777_02_secondary.jpg"
    finally:
        shutil.rmtree(out, ignore_errors=True)


def test_two_views_produce_two_secondaries():
    out = _tmp()
    try:
        v1 = save(synth_render(700, 1700, canvas=2200), out / "src" / "a.png")
        v2 = save(synth_render(650, 1700, canvas=2200), out / "src" / "b.png")
        job = Job(
            job_id="t4",
            model_id="a",
            views=[View(url=str(v1)), View(url=str(v2))],
            outputs=["main", "secondary"],
        )
        report = build_previews(job, SPEC, out)
        assert report.detected["type"] == "V"
        assert [r.role for r in report.results] == ["main", "secondary", "secondary"]
    finally:
        shutil.rmtree(out, ignore_errors=True)


def test_spheres_never_land_in_a_reserved_zone():
    out = _tmp()
    try:
        job = Job(
            job_id="t5",
            model_id="a",
            views=[View(url=str(MODEL))],
            spheres=[str(SPHERE)] * 4,
            outputs=["main"],
        )
        report = build_previews(job, SPEC, out)
        zones = SPEC.zone_rects()
        spheres = [p for p in report.results[0].placements if p.kind == "sphere"]
        assert spheres, "no spheres placed at all"
        for p in spheres:
            for name, z in zones.items():
                overlap = max(0, min(p.rect[2], z[2]) - max(p.rect[0], z[0])) * max(
                    0, min(p.rect[3], z[3]) - max(p.rect[1], z[1])
                )
                assert overlap == 0, f"sphere {p.index} intrudes into {name}"
    finally:
        shutil.rmtree(out, ignore_errors=True)


def test_secondary_never_carries_spheres():
    out = _tmp()
    try:
        job = Job(
            job_id="t6",
            model_id="a",
            views=[View(url=str(MODEL))],
            spheres=[str(SPHERE)] * 4,
            outputs=["main", "secondary"],
        )
        report = build_previews(job, SPEC, out)
        for r in report.results:
            if r.role == "secondary":
                assert not any(p.kind == "sphere" for p in r.placements)
    finally:
        shutil.rmtree(out, ignore_errors=True)


def test_broken_input_is_reported_not_raised():
    out = _tmp()
    try:
        bad = out / "src" / "broken.png"
        bad.parent.mkdir(parents=True, exist_ok=True)
        bad.write_bytes(b"not an image at all")
        job = Job(job_id="t7", model_id="a", views=[View(url=str(bad))], outputs=["main"])
        report = build_previews(job, SPEC, out)
        assert report.results == []
        assert any("no usable views" in w for w in report.warnings)
    finally:
        shutil.rmtree(out, ignore_errors=True)


def test_report_round_trips_to_json():
    out = _tmp()
    try:
        job = Job(job_id="t8", model_id="a", views=[View(url=str(MODEL))], outputs=["main"])
        report = build_previews(job, SPEC, out)
        report.save(out / "report.json")

        import json

        d = json.loads((out / "report.json").read_text(encoding="utf-8"))
        assert d["status"] in ("ok", "ok_with_warnings", "failed")
        assert d["spec_digest"] == SPEC.digest
        assert d["results"][0]["checks"]
    finally:
        shutil.rmtree(out, ignore_errors=True)
