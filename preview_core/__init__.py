"""preview_core - deterministic 3ddd preview composition.

Framework-agnostic on purpose: numpy + PIL + PyYAML only. Electron, FastAPI,
a node graph or a CLI all call the same functions, so the geometry can be
unit-tested and golden-tested without booting any of them.
"""

from .spec import Spec
from .models import Job, View, Report, ResultItem, Check, Placement, Transform, ViewAnalysis
from .geometry import analyse_view, classify, soft_mask, shadow_mask
from .layout import pick_preset, solve_layout, Preset
from .compose import compose, compose_closeup
from .occupancy import build_free_map
from .spheres import place_spheres
from .overlay import apply_overlays
from .closeup import propose_closeups
from .validate import validate, validate_file
from .export import export_jpeg
from .pipeline import build_previews

__version__ = "0.2.5"

__all__ = [
    "Spec", "Job", "View", "Report", "ResultItem", "Check", "Placement",
    "Transform", "ViewAnalysis", "Preset",
    "analyse_view", "classify", "soft_mask", "shadow_mask",
    "pick_preset", "solve_layout", "compose", "compose_closeup",
    "build_free_map", "place_spheres", "apply_overlays", "propose_closeups",
    "validate", "validate_file", "export_jpeg", "build_previews",
    "__version__",
]
