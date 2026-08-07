"""Reverse-engineer guide and zone coordinates from a template screenshot.

Э0 is blocked on the PSD. This tool is the interim answer: point it at a
screenshot of the template (121.png) and it prints the `guides` and `zones`
blocks for spec.yaml. When the real PSD arrives, run it once against a flat
export to confirm - or replace the numbers by hand and move on.

    python tools/calibrate_guides.py assets/121.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image

Rect = Tuple[int, int, int, int]

INK_LEVEL = 12       # a pixel counts as drawn at this distance from white
FULL_LENGTH = 0.80   # coverage above which a line is a guide, not a zone edge
DOWNSAMPLE = 4       # grouping resolution for zone outlines
EDGE_GUARD = 3       # px ignored at the artboard edge


def _detect_viewer_border(nonwhite: np.ndarray) -> Rect:
    """Screenshots usually carry a dark frame around the artboard.

    Takes the innermost strong line in each outer 10% band, so a two-pixel
    or double-stroked frame is stripped correctly.
    """
    h, w = nonwhite.shape
    col, row = nonwhite.mean(axis=0), nonwhite.mean(axis=1)
    strong = max(col.max(), row.max()) * 0.7
    bw, bh = max(1, w // 10), max(1, h // 10)

    left = [x for x in range(bw) if col[x] > strong]
    right = [x for x in range(w - bw, w) if col[x] > strong]
    top = [y for y in range(bh) if row[y] > strong]
    bottom = [y for y in range(h - bh, h) if row[y] > strong]
    return (
        max(left) + 1 if left else 0,
        max(top) + 1 if top else 0,
        min(right) if right else w,
        min(bottom) if bottom else h,
    )


def _line_positions(profile: np.ndarray, floor: float) -> List[Tuple[int, int, float]]:
    idx = np.nonzero(profile > floor)[0]
    groups: List[List[int]] = []
    for i in idx:
        if groups and i - groups[-1][-1] <= 2:
            groups[-1].append(int(i))
        else:
            groups.append([int(i)])
    return [(g[0], g[-1], float(profile[g].mean())) for g in groups]


def _centre(group: Tuple[int, int, float]) -> float:
    return (group[0] + group[1]) / 2.0


def calibrate(path: Path, verbose: bool = False) -> dict:
    rgb = np.asarray(Image.open(path).convert("RGB")).astype(np.int16)
    nonwhite = (255 - rgb.min(axis=2)).astype(np.float32)

    bx0, by0, bx1, by1 = _detect_viewer_border(nonwhite)
    inner = nonwhite[by0:by1, bx0:bx1]
    ih, iw = inner.shape
    if verbose:
        print(f"artboard: {iw}x{ih} at ({bx0},{by0})", file=sys.stderr)

    # A guide runs the full length of the artboard; a zone outline does not.
    # Ranking by coverage rather than by darkness is what separates them.
    ink = inner > INK_LEVEL
    vlines = _line_positions(ink.mean(axis=0), FULL_LENGTH)
    hlines = _line_positions(ink.mean(axis=1), FULL_LENGTH)
    if verbose:
        print(f"vertical lines: {vlines}", file=sys.stderr)
        print(f"horizontal lines: {hlines}", file=sys.stderr)

    # Guides are the outermost full-length lines on each side.
    def pick(lines, span: int) -> Tuple[Optional[float], Optional[float]]:
        if not lines:
            return None, None
        lo = min(lines, key=lambda g: _centre(g))
        hi = max(lines, key=lambda g: _centre(g))
        return _centre(lo), _centre(hi)

    lx, rx = pick(vlines, iw)
    ty, by = pick(hlines, ih)

    insets = [v / iw for v in (lx, iw - rx) if v is not None]
    insets += [v / ih for v in (ty, ih - by) if v is not None]
    inset = float(np.mean(insets)) if insets else 0.0
    spread = float(np.max(insets) - np.min(insets)) if insets else 0.0

    zones = _detect_zones(ink, vlines, hlines, iw, ih)

    return {
        "artboard": (iw, ih),
        "inset_ratio": round(inset, 4),
        "inset_px": round(inset * iw, 1),
        "asymmetry": round(spread, 4),
        "zones": zones,
    }


def _detect_zones(ink: np.ndarray, vlines, hlines, iw: int, ih: int) -> dict:
    """Everything left after the guides are erased is a zone outline.

    Colour-matching the rounded rectangles is fragile - they are drawn in
    several shades and heavily anti-aliased. Subtracting the full-length
    lines and taking what remains is not.
    """
    rest = ink.copy()
    # Anti-aliasing from the stripped viewer frame leaves a smear on the
    # outermost pixels; without this guard every zone bbox snaps to 0.0.
    rest[:EDGE_GUARD, :] = False
    rest[-EDGE_GUARD:, :] = False
    rest[:, :EDGE_GUARD] = False
    rest[:, -EDGE_GUARD:] = False
    for a, b, _ in vlines:
        rest[:, max(0, a - 1) : b + 2] = False
    for a, b, _ in hlines:
        rest[max(0, a - 1) : b + 2, :] = False
    if not rest.any():
        return {}

    # Group with a dilated copy, then measure on the original: closing the
    # stroke is needed to make one component, but it would inflate the rect
    # by the closing radius if we measured on it.
    closed = _close(rest, 3)
    labels = _label(_downsample(closed, DOWNSAMPLE))
    out: dict = {}
    for lid in range(1, int(labels.max()) + 1):
        region = np.zeros_like(rest)
        up = np.kron(labels == lid, np.ones((DOWNSAMPLE, DOWNSAMPLE), dtype=bool))
        region[: up.shape[0], : up.shape[1]] = up[: region.shape[0], : region.shape[1]]
        ys, xs = np.nonzero(rest & region)
        if xs.size < 20:
            continue
        rx0, ry0 = int(xs.min()) / iw, int(ys.min()) / ih
        rx1, ry1 = (int(xs.max()) + 1) / iw, (int(ys.max()) + 1) / ih
        if (rx1 - rx0) * (ry1 - ry0) < 0.004:  # stray marks, not a zone
            continue
        cx, cy = (rx0 + rx1) / 2, (ry0 + ry1) / 2
        name = ("T" if cy < 0.5 else "B") + ("L" if cx < 0.5 else "R")
        out[name] = [round(rx0, 4), round(ry0, 4), round(rx1, 4), round(ry1, 4)]
    return out


def _close(mask: np.ndarray, r: int) -> np.ndarray:
    """Dilate to bridge the gaps anti-aliasing leaves in a thin stroke."""
    out = mask.copy()
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            out |= np.roll(np.roll(mask, dy, axis=0), dx, axis=1)
    return out


def _downsample(mask: np.ndarray, factor: int = 4) -> np.ndarray:
    h, w = mask.shape
    h, w = h // factor * factor, w // factor * factor
    return mask[:h, :w].reshape(h // factor, factor, w // factor, factor).any(axis=(1, 3))


def _label(mask: np.ndarray) -> np.ndarray:
    """Connected components, 8-connected, iterative flood fill."""
    labels = np.zeros(mask.shape, dtype=np.int32)
    cur = 0
    h, w = mask.shape
    for sy in range(h):
        for sx in range(w):
            if not mask[sy, sx] or labels[sy, sx]:
                continue
            cur += 1
            stack = [(sy, sx)]
            labels[sy, sx] = cur
            while stack:
                y, x = stack.pop()
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not labels[ny, nx]:
                            labels[ny, nx] = cur
                            stack.append((ny, nx))
    return labels


def main(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="extract guide geometry from a template image")
    p.add_argument("image")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)

    res = calibrate(Path(args.image), args.verbose)
    iw, ih = res["artboard"]

    print(f"# calibrated from {args.image} ({iw}x{ih} artboard)")
    print("guides:")
    print(f"  inset_ratio: {res['inset_ratio']}    # {res['inset_px']}px on this artboard")
    if res["asymmetry"] > 0.002:
        print(f"  # WARNING: sides disagree by {res['asymmetry']:.2%} - guides are not symmetric")
    print("zones:")
    for name in ("TL", "BR", "BL", "TR"):
        if name in res["zones"]:
            print(f"  {name}: {res['zones'][name]}")
    missing = [n for n in ("TL", "BR", "BL") if n not in res["zones"]]
    if missing:
        print(f"  # not marked in this template: {', '.join(missing)} - confirm manually")
    return 0


if __name__ == "__main__":
    sys.exit(main())
