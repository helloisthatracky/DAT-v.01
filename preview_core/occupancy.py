"""Step 6: what is still free on the canvas.

Blocked = the model plus a clearance ring, plus every safe zone. Queries go
through a summed-area table, so "is this rectangle empty?" is O(1) and the
sphere search can afford to be exhaustive.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np

from .imaging import dilate
from .models import BBox
from .spec import Spec


@dataclass
class FreeMap:
    blocked: np.ndarray          # bool, True = occupied
    integral: np.ndarray         # (H+1, W+1) int32 summed-area table
    zones: Dict[str, BBox]

    @property
    def shape(self) -> Tuple[int, int]:
        return self.blocked.shape  # type: ignore[return-value]

    def is_free(self, box: BBox) -> bool:
        x0, y0, x1, y1 = box
        h, w = self.blocked.shape
        if x0 < 0 or y0 < 0 or x1 > w or y1 > h or x1 <= x0 or y1 <= y0:
            return False
        s = self.integral
        total = s[y1, x1] - s[y0, x1] - s[y1, x0] + s[y0, x0]
        return total == 0

    def free_ratio(self, box: Optional[BBox] = None) -> float:
        h, w = self.blocked.shape
        x0, y0, x1, y1 = box or (0, 0, w, h)
        area = max(1, (x1 - x0) * (y1 - y0))
        s = self.integral
        used = s[y1, x1] - s[y0, x1] - s[y1, x0] + s[y0, x0]
        return 1.0 - float(used) / area


def build_free_map(
    cover: np.ndarray, spec: Spec, include_zones: bool = True
) -> FreeMap:
    solid = cover > spec.mask.alpha_threshold
    clearance = spec.px(spec.spheres.clearance_ratio)
    blocked = dilate(solid, clearance)

    zones = spec.zone_rects()
    if include_zones:
        for x0, y0, x1, y1 in zones.values():
            blocked[y0:y1, x0:x1] = True

    integral = np.zeros(
        (blocked.shape[0] + 1, blocked.shape[1] + 1), dtype=np.int32
    )
    integral[1:, 1:] = np.cumsum(np.cumsum(blocked.astype(np.int32), axis=0), axis=1)
    return FreeMap(blocked=blocked, integral=integral, zones=zones)


def scan_free_rects(
    fm: FreeMap, region: BBox, size: Tuple[int, int], step: int
) -> Iterator[BBox]:
    """Yield every position of `size` that fits fully free inside `region`."""
    rx0, ry0, rx1, ry1 = region
    w, h = size
    if rx1 - rx0 < w or ry1 - ry0 < h:
        return
    ys = list(range(ry0, ry1 - h + 1, max(1, step)))
    xs = list(range(rx0, rx1 - w + 1, max(1, step)))
    # Always test the far edge too - the best slot is often flush against it.
    if ys and ys[-1] != ry1 - h:
        ys.append(ry1 - h)
    if xs and xs[-1] != rx1 - w:
        xs.append(rx1 - w)
    for y in ys:
        for x in xs:
            box = (x, y, x + w, y + h)
            if fm.is_free(box):
                yield box
