"""Typed access to spec.yaml.

Every rule of the 3ddd guide lives here as data. No module in preview_core
is allowed to hardcode a geometric constant.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml

Rect = Tuple[int, int, int, int]  # x0, y0, x1, y1 - inclusive-exclusive

SPEC_FILENAME = "spec.yaml"


def default_spec_path() -> Path:
    """Path to the spec shipped with the package.

    Resolved relative to this module, not to the working directory or the
    repository layout - an installed wheel has no repository around it, and
    the Electron bundle will ship exactly this package directory.
    """
    return Path(__file__).resolve().parent / SPEC_FILENAME


def _rect_from_ratio(r: List[float], w: int, h: int) -> Rect:
    return (round(r[0] * w), round(r[1] * h), round(r[2] * w), round(r[3] * h))


@dataclass(frozen=True)
class CanvasSpec:
    width: int
    height: int
    background: Tuple[int, int, int]

    @property
    def size(self) -> Tuple[int, int]:
        return (self.width, self.height)

    @property
    def short_side(self) -> int:
        return min(self.width, self.height)


@dataclass(frozen=True)
class GuideSpec:
    inset_ratio: float
    touch_tolerance_ratio: float
    max_overflow_px: int

    def box(self, canvas: CanvasSpec) -> Rect:
        mx = round(self.inset_ratio * canvas.width)
        my = round(self.inset_ratio * canvas.height)
        return (mx, my, canvas.width - mx, canvas.height - my)

    def tolerance(self, canvas: CanvasSpec) -> float:
        return self.touch_tolerance_ratio * canvas.short_side


@dataclass(frozen=True)
class ClassifySpec:
    h_threshold: float
    v_threshold: float


@dataclass(frozen=True)
class MaskSpec:
    alpha_threshold: float
    median_size: int
    bbox_coverage_eps: float


@dataclass(frozen=True)
class ShadowSpec:
    enabled: bool
    max_saturation: float
    min_luminance: float
    max_alpha: float
    below_centroid: bool


@dataclass(frozen=True)
class LayoutSpec:
    max_upscale: float
    resample: str
    gap_ratio: float
    y_anchor: float
    two_views_min_fill: float


@dataclass(frozen=True)
class SphereSpec:
    enabled: bool
    count: int
    diameter_ratio: float
    min_diameter_ratio: float
    gap_ratio: float
    clearance_ratio: float
    search_step_ratio: float
    priority: Dict[str, List[str]]


@dataclass(frozen=True)
class OverlaySlot:
    corner: str
    width_ratio: float
    margin_ratio: float
    enabled: bool = True


@dataclass(frozen=True)
class OverlaySpec:
    draw_3ddd_logo: bool
    logo_3ddd: OverlaySlot
    brand_logo: OverlaySlot


@dataclass(frozen=True)
class CloseupSpec:
    n: int
    min_free_area_ratio: float
    window_ratio: float
    step_ratio: float
    nms_iou: float
    bleed: bool
    entropy_weight: float
    gradient_weight: float


@dataclass(frozen=True)
class ExportSpec:
    format: str
    quality: int
    subsampling: int
    min_quality: int
    max_mb: float
    icc_srgb: bool
    strip_metadata: bool
    name_template: str


@dataclass(frozen=True)
class ValidateSpec:
    strict: bool
    bg_white_tolerance: int
    bg_sample_border_ratio: float
    min_canvas_side: int = 900


@dataclass(frozen=True)
class Spec:
    version: str
    canvas: CanvasSpec
    guides: GuideSpec
    zones: Dict[str, List[float]]
    classify: ClassifySpec
    mask: MaskSpec
    shadow: ShadowSpec
    layout: LayoutSpec
    spheres: SphereSpec
    overlays: OverlaySpec
    closeup: CloseupSpec
    export: ExportSpec
    validate: ValidateSpec
    digest: str = ""
    source_path: str = ""
    raw: Dict[str, Any] = field(default_factory=dict, repr=False)

    # -- derived geometry -------------------------------------------------
    def guide_box(self) -> Rect:
        return self.guides.box(self.canvas)

    def zone_rects(self) -> Dict[str, Rect]:
        return {
            name: _rect_from_ratio(r, self.canvas.width, self.canvas.height)
            for name, r in self.zones.items()
        }

    def for_canvas(self, width: int, height: int) -> "Spec":
        """Same rules, re-based on a different canvas size.

        Every rule in the guide is proportional, so a 1092px preview is
        checked against guides at 1092 * inset_ratio. Validating a foreign
        file against the 1500px geometry would flag every one of them.
        """
        if (width, height) == self.canvas.size:
            return self
        return replace(
            self, canvas=CanvasSpec(width, height, self.canvas.background)
        )

    def px(self, ratio: float, axis: str = "w") -> int:
        base = self.canvas.width if axis == "w" else self.canvas.height
        return round(ratio * base)

    # -- loading ----------------------------------------------------------
    @staticmethod
    def default() -> "Spec":
        """The spec that ships with this build of the package."""
        return Spec.load(default_spec_path())

    @staticmethod
    def load(path: str | Path) -> "Spec":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"spec not found: {path}\n"
                f"Omit --spec to use the one shipped with the package "
                f"({default_spec_path()})."
            )
        text = path.read_text(encoding="utf-8")
        return Spec.from_dict(
            yaml.safe_load(text),
            digest=hashlib.sha256(text.encode("utf-8")).hexdigest()[:16],
            source_path=str(path),
        )

    @staticmethod
    def from_dict(d: Dict[str, Any], digest: str = "", source_path: str = "") -> "Spec":
        ov = d["overlays"]
        return Spec(
            version=str(d.get("version", "0")),
            canvas=CanvasSpec(
                width=int(d["canvas"]["width"]),
                height=int(d["canvas"]["height"]),
                background=tuple(d["canvas"]["background"]),  # type: ignore[arg-type]
            ),
            guides=GuideSpec(**d["guides"]),
            zones=d["zones"],
            classify=ClassifySpec(**d["classify"]),
            mask=MaskSpec(**d["mask"]),
            shadow=ShadowSpec(**d["shadow"]),
            layout=LayoutSpec(**d["layout"]),
            spheres=SphereSpec(**d["spheres"]),
            overlays=OverlaySpec(
                draw_3ddd_logo=bool(ov["draw_3ddd_logo"]),
                logo_3ddd=OverlaySlot(**ov["logo_3ddd"]),
                brand_logo=OverlaySlot(**ov["brand_logo"]),
            ),
            closeup=CloseupSpec(**d["closeup"]),
            export=ExportSpec(**d["export"]),
            validate=ValidateSpec(**d["validate"]),
            digest=digest,
            source_path=source_path,
            raw=d,
        )
