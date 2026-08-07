"""Data contracts: job.json in, report.json out."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

BBox = Tuple[int, int, int, int]  # x0, y0, x1, y1 (x1/y1 exclusive)


# --------------------------------------------------------------------------
# Input
# --------------------------------------------------------------------------
@dataclass
class View:
    url: str
    angle: str = "unknown"


@dataclass
class Job:
    job_id: str
    model_id: str
    views: List[View]
    spheres: List[str] = field(default_factory=list)
    brand_logo: Optional[str] = None
    outputs: List[str] = field(default_factory=lambda: ["main", "secondary", "closeup"])
    overrides: Dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def load(path: str | Path) -> "Job":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return Job.from_dict(d)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Job":
        views = [View(**v) if isinstance(v, dict) else View(url=str(v)) for v in d["views"]]
        return Job(
            job_id=d.get("job_id", "local"),
            model_id=str(d.get("model_id", "model")),
            views=views,
            spheres=list(d.get("spheres") or []),
            brand_logo=d.get("brand_logo"),
            outputs=list(d.get("outputs") or ["main", "secondary", "closeup"]),
            overrides=dict(d.get("overrides") or {}),
        )

    def resolve(self, base: Path) -> "Job":
        """Make every relative path absolute against `base`."""

        def r(p: Optional[str]) -> Optional[str]:
            if p is None or "://" in p:
                return p
            q = Path(p)
            return str(q if q.is_absolute() else (base / q).resolve())

        return Job(
            job_id=self.job_id,
            model_id=self.model_id,
            views=[View(url=r(v.url) or v.url, angle=v.angle) for v in self.views],
            spheres=[r(s) or s for s in self.spheres],
            brand_logo=r(self.brand_logo),
            outputs=self.outputs,
            overrides=self.overrides,
        )


# --------------------------------------------------------------------------
# Intermediates
# --------------------------------------------------------------------------
@dataclass
class ViewAnalysis:
    """Everything the geometry stage learned about one render."""

    index: int
    path: str
    size: Tuple[int, int]
    solid_bbox: BBox           # shadow removed - drives the fit
    content_bbox: BBox         # shadow included - drives the crop
    shadow_ratio: float
    fill_ratio: float

    @property
    def solid_wh(self) -> Tuple[int, int]:
        x0, y0, x1, y1 = self.solid_bbox
        return (x1 - x0, y1 - y0)

    @property
    def ratio(self) -> float:
        w, h = self.solid_wh
        return w / h if h else 0.0


@dataclass
class Transform:
    """Affine placement of one view on the canvas (scale + translate only)."""

    view_index: int
    scale: float
    dst_solid: BBox            # where the solid bbox lands
    dst_content: BBox          # where the full crop lands (may exceed guides)


@dataclass
class Placement:
    kind: str                  # "sphere" | "logo_3ddd" | "brand_logo"
    index: int
    rect: BBox


@dataclass
class Check:
    id: str
    status: str                # pass | warn | fail
    msg: str = ""


@dataclass
class ResultItem:
    role: str
    index: int
    path: str
    checks: List[Check] = field(default_factory=list)
    placements: List[Placement] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def status(self) -> str:
        if any(c.status == "fail" for c in self.checks):
            return "fail"
        if any(c.status == "warn" for c in self.checks):
            return "warn"
        return "pass"


@dataclass
class Report:
    job_id: str
    model_id: str
    spec_version: str
    spec_digest: str
    detected: Dict[str, Any] = field(default_factory=dict)
    results: List[ResultItem] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def status(self) -> str:
        if any(r.status == "fail" for r in self.results):
            return "failed"
        if any(r.status == "warn" for r in self.results) or self.warnings:
            return "ok_with_warnings"
        return "ok"

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["status"] = self.status
        for r, src in zip(d["results"], self.results):
            r["status"] = src.status
        return d

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
