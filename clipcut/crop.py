from __future__ import annotations

from typing import TYPE_CHECKING

from .models import CropRect

if TYPE_CHECKING:
    from .models import ClipItem, ProjectManifest

MIN_CROP_RATIO = 0.05


def sanitize_crop_rect(
    x: float,
    y: float,
    w: float,
    h: float,
    *,
    min_ratio: float = MIN_CROP_RATIO,
) -> tuple[float, float, float, float]:
    """Clamp normalized crop rectangle into [0, 1] with minimum width/height."""
    mr = max(0.01, min(0.5, float(min_ratio)))
    sx = max(0.0, min(1.0 - mr, float(x)))
    sy = max(0.0, min(1.0 - mr, float(y)))
    sw = max(mr, min(1.0 - sx, float(w)))
    sh = max(mr, min(1.0 - sy, float(h)))
    return round(sx, 4), round(sy, 4), round(sw, 4), round(sh, 4)


def is_full_frame(x: float, y: float, w: float, h: float) -> bool:
    """Return True when the normalized crop rectangle covers essentially the entire frame."""
    return x <= 0.001 and y <= 0.001 and w >= 0.999 and h >= 0.999


def resolve_effective_crop(
    manifest: ProjectManifest,
    clip: ClipItem | None = None,
) -> CropRect | None:
    """Resolve effective normalized crop rect for a clip or project (None = uncropped)."""
    if clip is not None and clip.crop_override is not None:
        ov = clip.crop_override
        if ov.mode == "none":
            return None
        if ov.mode == "custom":
            sx, sy, sw, sh = sanitize_crop_rect(ov.x, ov.y, ov.w, ov.h)
            if is_full_frame(sx, sy, sw, sh):
                return None
            return CropRect(x=sx, y=sy, w=sw, h=sh)
    if manifest.crop.enabled:
        sx, sy, sw, sh = sanitize_crop_rect(
            manifest.crop.x,
            manifest.crop.y,
            manifest.crop.w,
            manifest.crop.h,
        )
        if is_full_frame(sx, sy, sw, sh):
            return None
        return CropRect(x=sx, y=sy, w=sw, h=sh)
    return None


def crop_rect_to_pixels(
    rect: CropRect | None,
    video_w: int,
    video_h: int,
) -> tuple[int, int, int, int] | None:
    """Convert normalized CropRect to even-aligned FFmpeg pixel bounds (x, y, w, h)."""
    if rect is None:
        return None
    vw = max(16, int(video_w) & ~1)
    vh = max(16, int(video_h) & ~1)
    sx, sy, sw, sh = sanitize_crop_rect(rect.x, rect.y, rect.w, rect.h)
    px = max(0, min(vw - 16, int(round(sx * vw)) & ~1))
    py = max(0, min(vh - 16, int(round(sy * vh)) & ~1))
    pw = max(16, min(vw - px, (int(round(sw * vw)) + 1) & ~1))
    ph = max(16, min(vh - py, (int(round(sh * vh)) + 1) & ~1))
    if px == 0 and py == 0 and pw >= vw and ph >= vh:
        return None
    return px, py, pw, ph


def crop_vf_expr(rect: CropRect | None, video_w: int, video_h: int) -> str:
    """Return FFmpeg crop=... filter expression or empty string when uncropped."""
    px_bounds = crop_rect_to_pixels(rect, video_w, video_h)
    if px_bounds is None:
        return ""
    px, py, pw, ph = px_bounds
    return f"crop={pw}:{ph}:{px}:{py}"


def crop_output_size(rect: CropRect | None, video_w: int, video_h: int) -> tuple[int, int]:
    """Return even-aligned output (width, height) after applying crop."""
    px_bounds = crop_rect_to_pixels(rect, video_w, video_h)
    if px_bounds is not None:
        return px_bounds[2], px_bounds[3]
    return max(16, int(video_w) & ~1), max(16, int(video_h) & ~1)


def crop_hash_payload(rect: CropRect | None) -> dict[str, float] | None:
    """Return canonical dict for render-hash calculation or None when uncropped."""
    if rect is None:
        return None
    return {
        "x": round(float(rect.x), 4),
        "y": round(float(rect.y), 4),
        "w": round(float(rect.w), 4),
        "h": round(float(rect.h), 4),
    }
