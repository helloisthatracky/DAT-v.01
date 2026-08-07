"""Image I/O and low-level pixel helpers. numpy + PIL only.

Convention used throughout preview_core:
    RGB  -> uint8 array, shape (H, W, 3)
    mask -> float32 array, shape (H, W), 0..1
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from PIL import Image, ImageCms, ImageFilter

Image.MAX_IMAGE_PIXELS = 300_000_000

RESAMPLE = {
    "lanczos": Image.Resampling.LANCZOS,
    "bicubic": Image.Resampling.BICUBIC,
    "bilinear": Image.Resampling.BILINEAR,
    "nearest": Image.Resampling.NEAREST,
}

SUPPORTED = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp", ".bmp"}


class InputError(ValueError):
    """Raised for unusable input files - caller turns it into a fail check."""


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
def load_rgba(path: str | Path) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Load an image as (rgb uint8, alpha float32|None), normalised to sRGB.

    An alpha channel is only returned when it actually carries information -
    a fully opaque channel is discarded so downstream code falls back to the
    white-background estimator.
    """
    p = Path(path)
    if p.suffix.lower() not in SUPPORTED:
        raise InputError(f"unsupported format: {p.suffix}")
    try:
        im = Image.open(p)
        im.load()
    except Exception as exc:  # noqa: BLE001 - broken file is a domain error
        raise InputError(f"cannot decode {p.name}: {exc}") from exc

    im = _to_srgb(im)

    alpha: Optional[np.ndarray] = None
    if im.mode in ("RGBA", "LA", "PA") or "transparency" in im.info:
        rgba = im.convert("RGBA")
        a = np.asarray(rgba)[:, :, 3].astype(np.float32) / 255.0
        if a.min() < 0.999:  # genuinely transparent somewhere
            alpha = a
            rgb = np.asarray(rgba)[:, :, :3].copy()
            return rgb, alpha
    rgb = np.asarray(im.convert("RGB")).copy()
    return rgb, alpha


def _to_srgb(im: Image.Image) -> Image.Image:
    """Convert an embedded ICC profile to sRGB; pass through when absent."""
    icc = im.info.get("icc_profile")
    if not icc:
        return im
    try:
        import io

        src = ImageCms.ImageCmsProfile(io.BytesIO(icc))
        dst = ImageCms.createProfile("sRGB")
        mode = "RGBA" if im.mode in ("RGBA", "LA", "PA") else "RGB"
        return ImageCms.profileToProfile(im, src, dst, outputMode=mode)  # type: ignore[return-value]
    except Exception:  # noqa: BLE001 - a bad profile must not kill the job
        return im


# --------------------------------------------------------------------------
# Masking helpers
# --------------------------------------------------------------------------
def white_alpha(rgb: np.ndarray) -> np.ndarray:
    """Soft coverage estimate for an object photographed/rendered on white.

    For a pixel over white:  P = C*a + 255*(1-a)  =>  a_hat = 1 - min(RGB)/255.
    Exact for black objects, conservative for light ones - which is all the
    geometry stage needs.
    """
    return 1.0 - rgb.min(axis=2).astype(np.float32) / 255.0


def median_filter(mask: np.ndarray, size: int) -> np.ndarray:
    if size < 3:
        return mask
    im = Image.fromarray((np.clip(mask, 0, 1) * 255).astype(np.uint8), mode="L")
    im = im.filter(ImageFilter.MedianFilter(size=size | 1))
    return np.asarray(im).astype(np.float32) / 255.0


def _dilate_axis(b: np.ndarray, r: int, axis: int) -> np.ndarray:
    """1-D binary dilation in O(1) per pixel via a prefix sum."""
    n = b.shape[axis]
    c = np.cumsum(b.astype(np.int32), axis=axis)
    zero = np.zeros_like(np.take(c, [0], axis=axis))
    c = np.concatenate([zero, c], axis=axis)  # length n+1
    idx = np.arange(n)
    hi = np.minimum(idx + r + 1, n)
    lo = np.maximum(idx - r, 0)
    return (np.take(c, hi, axis=axis) - np.take(c, lo, axis=axis)) > 0


def dilate(mask_bool: np.ndarray, radius_px: int) -> np.ndarray:
    """Binary dilation with a square structuring element.

    Separable prefix-sum implementation: a 30px radius on a 1500px canvas is
    milliseconds, where PIL's MaxFilter is seconds. The clearance ring around
    the model is recomputed for every preview, so this is on the hot path.
    """
    if radius_px <= 0:
        return mask_bool
    r = int(radius_px)
    return _dilate_axis(_dilate_axis(mask_bool, r, 0), r, 1)


def luminance(rgb: np.ndarray) -> np.ndarray:
    a = rgb.astype(np.float32) / 255.0
    return 0.2126 * a[:, :, 0] + 0.7152 * a[:, :, 1] + 0.0722 * a[:, :, 2]


def saturation(rgb: np.ndarray) -> np.ndarray:
    a = rgb.astype(np.float32)
    mx = a.max(axis=2)
    mn = a.min(axis=2)
    return np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-6), 0.0)


# --------------------------------------------------------------------------
# Geometry helpers
# --------------------------------------------------------------------------
def resize_rgb(rgb: np.ndarray, size: Tuple[int, int], resample: str) -> np.ndarray:
    im = Image.fromarray(rgb, mode="RGB")
    return np.asarray(im.resize(size, RESAMPLE[resample])).copy()


def resize_mask(mask: np.ndarray, size: Tuple[int, int], resample: str) -> np.ndarray:
    im = Image.fromarray((np.clip(mask, 0, 1) * 255).astype(np.uint8), mode="L")
    out = im.resize(size, RESAMPLE[resample])
    return np.asarray(out).astype(np.float32) / 255.0


def blank_canvas(w: int, h: int, bg: Tuple[int, int, int]) -> np.ndarray:
    c = np.empty((h, w, 3), dtype=np.uint8)
    c[:, :] = np.array(bg, dtype=np.uint8)
    return c


def paste_darken(canvas: np.ndarray, patch: np.ndarray, x: int, y: int) -> None:
    """Composite `patch` onto `canvas` at (x, y) with a per-channel darken blend.

    On a white canvas darken is a bit-exact pixel transfer, so no matting and
    no halo. It also makes overlapping views degrade gracefully instead of
    one white bounding box erasing its neighbour.
    """
    ch, cw = canvas.shape[:2]
    ph, pw = patch.shape[:2]
    sx0, sy0 = max(0, -x), max(0, -y)
    dx0, dy0 = max(0, x), max(0, y)
    dx1, dy1 = min(cw, x + pw), min(ch, y + ph)
    if dx1 <= dx0 or dy1 <= dy0:
        return
    sub = patch[sy0 : sy0 + (dy1 - dy0), sx0 : sx0 + (dx1 - dx0)]
    dst = canvas[dy0:dy1, dx0:dx1]
    np.minimum(dst, sub, out=dst)


def paste_alpha(
    canvas: np.ndarray, patch: np.ndarray, alpha: np.ndarray, x: int, y: int
) -> None:
    """Standard source-over composite, used for overlays and spheres."""
    ch, cw = canvas.shape[:2]
    ph, pw = patch.shape[:2]
    sx0, sy0 = max(0, -x), max(0, -y)
    dx0, dy0 = max(0, x), max(0, y)
    dx1, dy1 = min(cw, x + pw), min(ch, y + ph)
    if dx1 <= dx0 or dy1 <= dy0:
        return
    s = patch[sy0 : sy0 + (dy1 - dy0), sx0 : sx0 + (dx1 - dx0)].astype(np.float32)
    a = alpha[sy0 : sy0 + (dy1 - dy0), sx0 : sx0 + (dx1 - dx0)].astype(np.float32)[..., None]
    dst = canvas[dy0:dy1, dx0:dx1].astype(np.float32)
    canvas[dy0:dy1, dx0:dx1] = np.round(s * a + dst * (1.0 - a)).astype(np.uint8)


def crop(arr: np.ndarray, box: Tuple[int, int, int, int]) -> np.ndarray:
    x0, y0, x1, y1 = box
    return arr[y0:y1, x0:x1]
