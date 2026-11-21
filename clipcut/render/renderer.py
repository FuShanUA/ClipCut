from __future__ import annotations

import functools
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from ..adapters.ingest import _find_best_info_json, extract_published_date
from ..assets_gen import ensure_default_assets
from ..config import load_local_config
from ..models import ClipItem, ProjectManifest
from ..pipeline.verify_and_plan import (
    compute_clip_render_hash,
    compute_compilation_render_hash,
    compute_remix_render_hash,
)
from ..receipts import ProjectStore
from .frame_selector import (
    FaceBox,
    SelectedFrame,
    annotate_faces_with_guest,
    build_hero_portrait_pool,
    compose_cinema_fit_width_frame,
    crop_frame_face_safe,
    crop_pip_portrait,
    detect_faces_batch,
    load_guest_profile,
    score_candidate_frame,
    select_best_frame_for_interval,
)

COVER_TEMPLATE_VERSION = "v2.3-tiffany-guest"
REMIX_CARD_TEMPLATE_VERSION = "v2-vision-pip"
_AUTOSUB_COVER_UTILS_PATH: Path | None = None

CATEGORY_LABELS_ZH = {
    "key_dialogue": "重点对话",
    "exciting_dialogue": "精彩对话",
    "hot_topic": "中国热点",
}

BEAT_LABELS_ZH = {
    "punchline": "论断",
    "qa_argument": "问答论证",
    "case_demo": "案例演示",
}


def should_insert_title_card(card_mode: str, idx: int, group_key: str, last_group: str | None) -> bool:
    if card_mode == "every":
        return True
    if card_mode == "group":
        return idx == 0 or group_key != last_group
    return False


def remix_item_group_key(item: Any) -> str:
    return (getattr(item, "group_label", "") or "").strip() or (getattr(item, "theme_id", "") or "").strip() or item.remix_id


def _fmt_dur(sec: float) -> str:
    s = max(0, int(round(sec)))
    h, rem = divmod(s, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def _clean_file_token(text: str, max_len: int) -> str:
    cleaned = re.sub(r'[<>:"/\\|?*\x00-\x1f,，。！？、；："\'‘’“”【】《》·—()（）\[\]]+', " ", str(text or ""))
    cleaned = re.sub(r"\s+", "-", cleaned).strip("-_")
    cleaned = re.sub(r"-{2,}", "-", cleaned)
    return cleaned[:max_len].strip("-_")


def _fmt_ts_token(sec: float) -> str:
    s = max(0, int(round(sec)))
    h, rem = divmod(s, 3600)
    m, s = divmod(rem, 60)
    return f"{h}h{m:02d}m{s:02d}s" if h else f"{m:02d}m{s:02d}s"


def build_clip_filename(clip: ClipItem, manifest: ProjectManifest, published_date: str = "") -> str:
    ch = _clean_file_token(manifest.metadata.channel_zh or manifest.metadata.channel or "", 24)
    title = _clean_file_token(clip.title, 36) or _clean_file_token(clip.slug, 36) or f"clip{clip.index:02d}"
    dt = _clean_file_token(published_date or manifest.metadata.published_date or "", 12)
    ts_range = f"{_fmt_ts_token(clip.start_s)}-{_fmt_ts_token(clip.end_s)}"
    parts = [f"{clip.index:02d}"] + [p for p in (ch, title, dt, ts_range) if p]
    return "_".join(parts) + ".mp4"


def _estimate_remix_duration_s(manifest: ProjectManifest) -> float:
    enabled_items = [it for it in sorted(manifest.remix.items, key=lambda x: x.order) if it.enabled]
    speech_s = sum(max(0.0, it.end_s - it.start_s) for it in enabled_items)
    last_gk: str | None = None
    cards = 0
    for idx, it in enumerate(enabled_items):
        gk = remix_item_group_key(it)
        if should_insert_title_card(manifest.remix.card_mode, idx, gk, last_gk):
            cards += 1
        last_gk = gk
    card_dur = 0.0 if manifest.remix.card_mode == "none" else float(manifest.remix.card_duration_s or 1.4)
    return speech_s + cards * card_dur


def build_remix_filename(manifest: ProjectManifest, published_date: str = "", duration_s: float | None = None) -> str:
    ch = _clean_file_token(manifest.metadata.channel_zh or manifest.metadata.channel or "", 24)
    title = _clean_file_token(manifest.remix.title or manifest.metadata.original_theme_zh or "高光集锦", 36)
    dt = _clean_file_token(published_date or manifest.metadata.published_date or "", 12)
    dur_val = duration_s if (duration_s is not None and duration_s > 0) else _estimate_remix_duration_s(manifest)
    parts = ["00", "Remix"] + [p for p in (ch, title, dt, _fmt_ts_token(dur_val)) if p]
    return "_".join(parts) + ".mp4"


def build_compilation_filename(manifest: ProjectManifest, published_date: str = "", duration_s: float | None = None) -> str:
    ch = _clean_file_token(manifest.metadata.channel_zh or manifest.metadata.channel or "", 24)
    title = _clean_file_token(manifest.compilation.title or "切片精选集", 36)
    dt = _clean_file_token(published_date or manifest.metadata.published_date or "", 12)
    enabled_clips = [c for c in manifest.clips if c.enabled]
    card_dur = float(manifest.compilation.card_duration_s or 1.8)
    dur_val = duration_s if (duration_s is not None and duration_s > 0) else (
        sum(max(0.0, c.end_s - c.start_s) for c in enabled_clips) + len(enabled_clips) * card_dur
    )
    parts = ["00", "Compilation"] + [p for p in (ch, title, dt, _fmt_ts_token(dur_val)) if p]
    return "_".join(parts) + ".mp4"


def _migrate_cached_media(old_file: str | None, new_mp4: Path, new_cover: Path) -> bool:
    """In-place rename existing rendered MP4 + companion JPG when filename convention updates."""
    if not old_file:
        return False
    old_mp4 = Path(old_file)
    if not old_mp4.exists() or old_mp4.stat().st_size <= 1024:
        return False
    if old_mp4.resolve() == new_mp4.resolve():
        return True
    new_mp4.parent.mkdir(parents=True, exist_ok=True)
    old_mp4.replace(new_mp4)
    old_cover = old_mp4.with_suffix(".jpg")
    if old_cover.exists() and not new_cover.exists():
        old_cover.replace(new_cover)
    return True


def _safe_filename(index: int, title: str, slug: str) -> str:
    cn_part = _clean_file_token(title, 22)
    slug_part = re.sub(r"[^a-z0-9-]+", "-", (slug or "clip").lower()).strip("-")[:36] or "clip"
    return f"{index:02d}_{cn_part}_{slug_part}.mp4"


def _has_videotoolbox() -> bool:
    try:
        res = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True, timeout=5)
        return "h264_videotoolbox" in res.stdout
    except Exception:
        return False


def _load_cjk_font(
    size: int,
    weight: str = "regular",
    bold: bool = False,
) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    if bold and weight == "regular":
        weight = "bold"
    user_fonts = Path.home() / "Library" / "Fonts"
    weight_map: dict[str, list[tuple[str, int]]] = {
        "heavy": [
            (str(user_fonts / "SourceHanSansSC-Heavy.otf"), 0),
            (str(user_fonts / "SourceHanSansSC-Bold.otf"), 0),
            ("/System/Library/Fonts/Hiragino Sans GB.ttc", 2),
            ("/System/Library/Fonts/STHeiti Medium.ttc", 1),
        ],
        "bold": [
            (str(user_fonts / "SourceHanSansSC-Bold.otf"), 0),
            (str(user_fonts / "SourceHanSansSC-Medium.otf"), 0),
            ("/System/Library/Fonts/Hiragino Sans GB.ttc", 2),
            ("/System/Library/Fonts/STHeiti Medium.ttc", 1),
        ],
        "medium": [
            (str(user_fonts / "SourceHanSansSC-Medium.otf"), 0),
            (str(user_fonts / "SourceHanSansSC-Normal.otf"), 0),
            ("/System/Library/Fonts/Hiragino Sans GB.ttc", 0),
            ("/System/Library/Fonts/STHeiti Medium.ttc", 1),
        ],
        "regular": [
            (str(user_fonts / "SourceHanSansSC-Regular.otf"), 0),
            (str(user_fonts / "SourceHanSansSC-Normal.otf"), 0),
            ("/System/Library/Fonts/Hiragino Sans GB.ttc", 0),
            ("/System/Library/Fonts/STHeiti Light.ttc", 1),
            ("/Library/Fonts/Arial Unicode.ttf", 0),
        ],
    }
    for fp, idx in weight_map.get(weight, weight_map["regular"]):
        p = Path(fp)
        if p.exists():
            try:
                return ImageFont.truetype(str(p), size=size, index=idx)
            except Exception:
                try:
                    return ImageFont.truetype(str(p), size=size)
                except Exception:
                    continue
    return ImageFont.load_default()


def _ensure_published_date(manifest: ProjectManifest, store: ProjectStore) -> str:
    """Backfill broadcast date from sibling .info.json or cached fetched_info.json if absent."""
    if manifest.metadata.published_date:
        return manifest.metadata.published_date
    vpath = Path(manifest.source.render_video)
    candidates: list[Path] = []
    if vpath.parent.exists():
        found = _find_best_info_json(vpath, vpath.parent)
        if found:
            candidates.append(found)
    fetched = store.work_dir / "fetched_info.json"
    if fetched.exists():
        candidates.append(fetched)
    for cand in candidates:
        try:
            raw = json.loads(cand.read_text(encoding="utf-8"))
            dt = extract_published_date(raw)
            if dt:
                manifest.metadata.published_date = dt
                return dt
        except Exception:
            continue
    return ""


def _build_cover_meta_parts(manifest: ProjectManifest, published_date: str) -> list[str]:
    guest = (manifest.metadata.guest_zh or "").strip()
    channel = (manifest.metadata.channel_zh or manifest.metadata.channel or "").strip()
    return [p for p in (guest, channel, published_date.strip()) if p]


def compute_cover_hash(
    *,
    key: str,
    title: str,
    categories: list[str],
    start_s: float,
    end_s: float,
    manifest: ProjectManifest,
    cfg: dict[str, Any],
    published_date: str,
) -> str:
    c_cfg = cfg.get("cover", {})
    payload = {
        "ver": COVER_TEMPLATE_VERSION,
        "key": key,
        "title": title,
        "categories": categories,
        "start_s": round(start_s, 3),
        "end_s": round(end_s, 3),
        "analysis_video": manifest.source.analysis_video,
        "guest_zh": manifest.metadata.guest_zh,
        "channel_zh": manifest.metadata.channel_zh or manifest.metadata.channel,
        "published_date": published_date,
        "style": c_cfg.get("style", "natural"),
        "prefer_two_shot": bool(c_cfg.get("prefer_two_shot", True)),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()[:16]


def _wrap_text(draw: ImageDraw.ImageDraw, text: str, font: Any, max_width: int) -> list[str]:
    lines: list[str] = []
    cur = ""
    for ch in text:
        test = cur + ch
        bbox = draw.textbbox((0, 0), test, font=font)
        if bbox[2] - bbox[0] <= max_width or not cur:
            cur = test
        else:
            lines.append(cur)
            cur = ch
    if cur:
        lines.append(cur)
    return lines


def _run_ffmpeg_encode(cmd_vt: list[str], cmd_x264: list[str], use_vt: bool) -> tuple[bool, str, str]:
    if use_vt:
        res = subprocess.run(cmd_vt, capture_output=True, text=True)
        if res.returncode == 0:
            return True, "h264_videotoolbox", ""
    res2 = subprocess.run(cmd_x264, capture_output=True, text=True)
    if res2.returncode == 0:
        return True, "libx264", ""
    return False, "failed", res2.stderr[-600:]


def _render_single_clip(
    source_video: Path,
    clip: ClipItem,
    out_path: Path,
    fade_ms: int,
    use_vt: bool,
) -> tuple[bool, str, str]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    start_s = max(0.0, clip.start_s)
    dur = max(0.5, clip.end_s - start_s)
    fade_s = min(fade_ms / 1000.0, dur * 0.15)
    fade_out_st = max(0.0, dur - fade_s)
    af = f"afade=t=in:st=0:d={fade_s:.3f},afade=t=out:st={fade_out_st:.3f}:d={fade_s:.3f}"

    base = [
        "ffmpeg",
        "-y",
        "-v",
        "error",
        "-ss",
        f"{start_s:.3f}",
        "-i",
        str(source_video),
        "-t",
        f"{dur:.3f}",
        "-af",
        af,
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-ar",
        "44100",
        "-ac",
        "2",
        "-movflags",
        "+faststart",
    ]
    cmd_vt = base + ["-c:v", "h264_videotoolbox", "-q:v", "60", str(out_path)]
    cmd_x264 = base + ["-c:v", "libx264", "-preset", "fast", "-crf", "18", str(out_path)]
    return _run_ffmpeg_encode(cmd_vt, cmd_x264, use_vt)


def _draw_top_badges(
    img: Image.Image,
    badge_text: str,
    dur_text: str,
    faces: list[FaceBox] | None = None,
    theme: str = "tiffany",
) -> None:
    w, h = img.size
    min_dim = min(w, h)
    draw = ImageDraw.Draw(img)
    pad_x = int(w * 0.042)
    by = int(h * 0.034)
    badge_font = _load_cjk_font(max(20, int(min_dim * 0.029)), weight="bold")

    b_box = draw.textbbox((0, 0), badge_text, font=badge_font)
    bw, bh = b_box[2] - b_box[0], b_box[3] - b_box[1]
    bx = pad_x
    pad_bx = max(14, int(min_dim * 0.018))
    pad_by = max(8, int(min_dim * 0.011))
    pill_h = bh + pad_by * 2
    radius = max(10, int(min_dim * 0.012))

    if theme == "tiffany":
        draw.rounded_rectangle(
            [bx + 3, by + 4, bx + bw + pad_bx * 2 + 3, by + pill_h + 4],
            radius=radius,
            fill=(0, 0, 0, 140),
        )
        draw.rounded_rectangle(
            [bx, by, bx + bw + pad_bx * 2, by + pill_h],
            radius=radius,
            fill=(12, 174, 168, 246),
            outline=(244, 255, 254, 250),
            width=3,
        )
        draw.text((bx + pad_bx - b_box[0] + 1, by + pad_by - b_box[1] + 1), badge_text, font=badge_font, fill=(4, 48, 52, 180))
        draw.text((bx + pad_bx - b_box[0], by + pad_by - b_box[1]), badge_text, font=badge_font, fill=(255, 255, 255, 255))
    else:
        draw.rounded_rectangle(
            [bx, by, bx + bw + pad_bx * 2, by + pill_h],
            radius=radius,
            fill=(217, 119, 6, 242),
        )
        draw.text((bx + pad_bx - b_box[0], by + pad_by - b_box[1]), badge_text, font=badge_font, fill=(255, 255, 255, 255))

    if not dur_text:
        return
    d_box = draw.textbbox((0, 0), dur_text, font=badge_font)
    dw = d_box[2] - d_box[0]
    tr_overlap = any((f.x + f.w) > 0.78 and f.y < 0.16 for f in (faces or []))
    dx = (bx + bw + pad_bx * 2 + 12) if tr_overlap else (w - pad_x - dw - pad_bx * 2)
    if theme == "tiffany":
        draw.rounded_rectangle(
            [dx + 3, by + 4, dx + dw + pad_bx * 2 + 3, by + pill_h + 4],
            radius=radius,
            fill=(0, 0, 0, 140),
        )
        draw.rounded_rectangle(
            [dx, by, dx + dw + pad_bx * 2, by + pill_h],
            radius=radius,
            fill=(7, 36, 42, 236),
            outline=(32, 212, 204, 255),
            width=3,
        )
        draw.text((dx + pad_bx - d_box[0], by + pad_by - d_box[1]), dur_text, font=badge_font, fill=(255, 255, 255, 255))
    else:
        draw.rounded_rectangle(
            [dx, by, dx + dw + pad_bx * 2, by + pill_h],
            radius=radius,
            fill=(8, 12, 20, 228),
            outline=(245, 158, 11, 200),
            width=2,
        )
        draw.text((dx + pad_bx - d_box[0], by + pad_by - d_box[1]), dur_text, font=badge_font, fill=(248, 250, 252, 255))


def _draw_frosted_glass_box(
    base_img: Image.Image,
    *,
    title: str,
    meta_parts: list[str],
    faces: list[FaceBox] | None = None,
    bottom_margin_ratio: float = 0.032,
    min_width_ratio: float = 0.64,
    max_width_ratio: float = 0.92,
    bg_alpha: int | None = None,
    border_width: int | None = None,
    title_color: tuple[int, int, int, int] | None = None,
    max_title_ratio: float | None = None,
    theme: str = "tiffany",
) -> Image.Image:
    """
    Composite a centered, high-contrast elevated card near the bottom.
    Supports luminous Tiffany Blue ('tiffany') cover theme with thick white border & layered shadow,
    plus dark charcoal ('obsidian') theme for in-video editorial PiP title cards.
    """
    w, h = base_img.size
    min_dim = min(w, h)
    ar = w / max(1, h)
    probe_draw = ImageDraw.Draw(base_img)

    is_tiffany = theme == "tiffany"
    stroke_w = max(2, int(min_dim * 0.0024)) if is_tiffany else 2
    stroke_fill = (5, 44, 48, 240) if is_tiffany else (0, 0, 0, 225)
    eff_title_color = title_color or ((255, 255, 255, 255) if is_tiffany else (254, 232, 84, 255))

    if ar >= 1.5:
        max_lines = 2
        max_width_ratio = max(max_width_ratio, 0.92)
        min_width_ratio = max(min_width_ratio, 0.64)
        init_ratio = 0.084 if len(title) <= 12 else 0.074
        if max_title_ratio is not None:
            init_ratio = min(init_ratio, max_title_ratio)
        title_font_sz = int(h * init_ratio)
    elif ar >= 1.0:
        max_lines = 2
        max_width_ratio = max(max_width_ratio, 0.92)
        min_width_ratio = max(min_width_ratio, 0.68)
        init_ratio = 0.078 if len(title) <= 12 else 0.068
        if max_title_ratio is not None:
            init_ratio = min(init_ratio, max_title_ratio)
        title_font_sz = int(w * init_ratio)
    else:
        max_lines = 3
        max_width_ratio = max(max_width_ratio, 0.92)
        min_width_ratio = max(min_width_ratio, 0.78)
        init_ratio = 0.090 if len(title) <= 12 else 0.078
        if max_title_ratio is not None:
            init_ratio = min(init_ratio, max_title_ratio)
        title_font_sz = int(w * init_ratio)

    min_font_sz = max(24, int(min_dim * 0.036))
    title_font_sz = max(min_font_sz, title_font_sz)
    pad_x_inner = max(36, int(w * 0.038))
    max_text_w = max(120, int(w * max_width_ratio) - pad_x_inner * 2)

    title_font = _load_cjk_font(title_font_sz, weight="heavy")
    while len(_wrap_text(probe_draw, title, title_font, max_text_w)) > max_lines and title_font_sz > min_font_sz:
        title_font_sz = max(min_font_sz, title_font_sz - 3)
        title_font = _load_cjk_font(title_font_sz, weight="heavy")
    lines = _wrap_text(probe_draw, title, title_font, max_text_w)

    meta_text = "  ·  ".join(meta_parts)
    meta_font_sz = max(18, int(min_dim * 0.030))
    meta_font = _load_cjk_font(meta_font_sz, weight="medium")
    while meta_text and meta_font_sz > 15:
        mb = probe_draw.textbbox((0, 0), meta_text, font=meta_font)
        if (mb[2] - mb[0]) <= max_text_w:
            break
        meta_font_sz -= 2
        meta_font = _load_cjk_font(meta_font_sz, weight="medium")

    def _measure_box(t_sz: int, lns: list[str], tf: Any, compact: bool = False) -> tuple[int, int, int, int, int, int]:
        max_lw = 0
        for ln in lns:
            lb = probe_draw.textbbox((0, 0), ln, font=tf, stroke_width=stroke_w)
            max_lw = max(max_lw, lb[2] - lb[0])
        mw = 0
        mh = 0
        if meta_text:
            mb2 = probe_draw.textbbox((0, 0), meta_text, font=meta_font)
            mw = mb2[2] - mb2[0]
            mh = max(meta_font_sz + 8, mb2[3] - mb2[1] + 4)
        pad_top = int(min_dim * (0.024 if compact else 0.032))
        pad_bot = int(min_dim * (0.020 if compact else 0.026))
        line_h = int(t_sz * (1.18 if compact else 1.24))
        div_gap = int(min_dim * (0.018 if compact else 0.024)) if meta_text else 0
        bw = max(int(w * min_width_ratio), min(int(w * max_width_ratio), max(max_lw, mw) + pad_x_inner * 2))
        bh = pad_top + len(lns) * line_h + div_gap + mh + pad_bot
        return bw, bh, pad_top, line_h, div_gap, mh

    box_w, box_h, pad_top, line_h, div_gap, meta_h = _measure_box(title_font_sz, lines, title_font, compact=False)
    bottom_limit_px = h - max(12, int(h * bottom_margin_ratio))
    by0 = bottom_limit_px - box_h

    max_face_bottom_px = max((int(round(f.bottom * h)) for f in (faces or [])), default=0)
    if max_face_bottom_px > 0 and by0 < max_face_bottom_px + 10:
        bottom_limit_px = h - max(10, int(h * 0.016))
        by0 = bottom_limit_px - box_h
        if by0 < max_face_bottom_px + 8:
            title_font_sz = max(min_font_sz, int(title_font_sz * 0.88))
            title_font = _load_cjk_font(title_font_sz, weight="heavy")
            lines = _wrap_text(probe_draw, title, title_font, max_text_w)
            box_w, box_h, pad_top, line_h, div_gap, meta_h = _measure_box(title_font_sz, lines, title_font, compact=True)
            by0 = max(max_face_bottom_px + 6, min(bottom_limit_px - box_h, h - box_h - 10))

    bx0 = max(0, (w - box_w) // 2)
    bx1 = min(w - 1, bx0 + box_w)
    by1 = min(h - 1, by0 + box_h)
    by0 = max(0, by0)
    box_w = max(2, bx1 - bx0)
    box_h = max(2, by1 - by0)
    radius = max(14, int(min_dim * 0.020))

    mask = Image.new("L", (box_w, box_h), 0)
    ImageDraw.Draw(mask).rounded_rectangle([0, 0, box_w, box_h], radius=radius, fill=255)

    patch = base_img.crop((bx0, by0, bx1, by1)).filter(ImageFilter.GaussianBlur(radius=8 if is_tiffany else 10))
    if is_tiffany:
        # Luminous Tiffany Blue vertical gradient fill + deep-teal recessed bottom strip
        yg = np.linspace(0.0, 1.0, box_h, dtype=np.float32)[:, None]
        grad_arr = np.zeros((box_h, box_w, 4), dtype=np.uint8)
        top_a = float(240 if bg_alpha is None else bg_alpha)
        bot_a = float(min(252, int(top_a + 6)))
        grad_arr[:, :, 0] = np.clip(24.0 * (1.0 - yg) + 10.0 * yg, 0, 255).astype(np.uint8)
        grad_arr[:, :, 1] = np.clip(198.0 * (1.0 - yg) + 154.0 * yg, 0, 255).astype(np.uint8)
        grad_arr[:, :, 2] = np.clip(190.0 * (1.0 - yg) + 152.0 * yg, 0, 255).astype(np.uint8)
        grad_arr[:, :, 3] = np.clip(top_a * (1.0 - yg) + bot_a * yg, 0, 255).astype(np.uint8)
        tint_layer = Image.fromarray(grad_arr, mode="RGBA")
        frosted = Image.alpha_composite(patch, tint_layer)

        # Layered prominent drop shadows (wide ambient shadow + directional crisp offset shadow)
        amb_shadow = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        ImageDraw.Draw(amb_shadow).rounded_rectangle(
            [max(0, bx0 - 6), max(0, by0 - 2), min(w - 1, bx1 + 14), min(h - 1, by1 + 18)],
            radius=radius + 6,
            fill=(0, 0, 0, 175),
        )
        amb_shadow = amb_shadow.filter(ImageFilter.GaussianBlur(radius=22))
        out = Image.alpha_composite(base_img.copy(), amb_shadow)

        off_x = max(8, int(min_dim * 0.010))
        off_y = max(10, int(min_dim * 0.012))
        off_shadow = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        ImageDraw.Draw(off_shadow).rounded_rectangle(
            [max(0, bx0 + off_x), max(0, by0 + off_y), min(w - 1, bx1 + off_x), min(h - 1, by1 + off_y)],
            radius=radius,
            fill=(3, 14, 18, 225),
        )
        off_shadow = off_shadow.filter(ImageFilter.GaussianBlur(radius=6))
        out = Image.alpha_composite(out, off_shadow)
    else:
        eff_alpha = 232 if bg_alpha is None else bg_alpha
        tint = Image.new("RGBA", (box_w, box_h), (5, 7, 12, eff_alpha))
        frosted = Image.alpha_composite(patch, tint)

        shadow_layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        ImageDraw.Draw(shadow_layer).rounded_rectangle(
            [max(0, bx0 - 6), max(0, by0 - 2), min(w - 1, bx1 + 6), min(h - 1, by1 + 14)],
            radius=radius + 4,
            fill=(0, 0, 0, 190),
        )
        shadow_layer = shadow_layer.filter(ImageFilter.GaussianBlur(radius=18))
        out = Image.alpha_composite(base_img.copy(), shadow_layer)

    div_y_abs = by0 + pad_top + len(lines) * line_h + max(4, div_gap // 2 - 2)
    if is_tiffany and meta_parts:
        strip_y0 = max(0, min(box_h - 1, div_y_abs - by0))
        strip_layer = Image.new("RGBA", (box_w, box_h), (0, 0, 0, 0))
        ImageDraw.Draw(strip_layer).rectangle([0, strip_y0, box_w, box_h], fill=(5, 62, 66, 165))
        frosted = Image.alpha_composite(frosted, strip_layer)

    out.paste(frosted, (bx0, by0), mask)
    draw = ImageDraw.Draw(out)

    # Frame outline + centered top accent bar
    eff_border_w = border_width if border_width is not None else (
        max(5, int(min_dim * 0.0056)) if is_tiffany else max(3, int(min_dim * 0.0038))
    )
    outline_col = (244, 255, 254, 255) if is_tiffany else (245, 158, 11, 255)
    draw.rounded_rectangle(
        [bx0, by0, bx1, by1],
        radius=radius,
        outline=outline_col,
        width=eff_border_w,
    )
    pill_w = max(56, int(box_w * 0.08))
    pill_h = max(4, int(min_dim * 0.0048))
    pill_fill = (255, 255, 255, 235) if is_tiffany else (253, 224, 71, 255)
    draw.rounded_rectangle(
        [w // 2 - pill_w // 2, by0 + 10, w // 2 + pill_w // 2, by0 + 10 + pill_h],
        radius=pill_h // 2,
        fill=pill_fill,
    )

    # Centered title lines
    y_cursor = by0 + pad_top
    shadow_col = (3, 28, 32, 200) if is_tiffany else (0, 0, 0, 210)
    sh_dx, sh_dy = (4, 4) if is_tiffany else (3, 3)
    for ln in lines:
        lb = draw.textbbox((0, 0), ln, font=title_font, stroke_width=stroke_w)
        lw = lb[2] - lb[0]
        tx = (w - lw) // 2 - lb[0]
        ty = y_cursor - lb[1] + max(0, (line_h - (lb[3] - lb[1])) // 2 - 2)
        draw.text((tx + sh_dx, ty + sh_dy), ln, font=title_font, fill=shadow_col, stroke_width=stroke_w, stroke_fill=shadow_col)
        draw.text((tx, ty), ln, font=title_font, fill=eff_title_color, stroke_width=stroke_w, stroke_fill=stroke_fill)
        y_cursor += line_h

    # Divider + centered metadata parts
    if meta_parts:
        div_y = y_cursor + max(4, div_gap // 2 - 2)
        div_margin = max(28, int(box_w * 0.055))
        div_col = (235, 255, 253, 215) if is_tiffany else (245, 158, 11, 155)
        draw.line([(bx0 + div_margin, div_y), (bx1 - div_margin, div_y)], fill=div_col, width=2 if is_tiffany else max(1, eff_border_w // 2))
        meta_y = y_cursor + div_gap
        mb_full = draw.textbbox((0, 0), meta_text, font=meta_font)
        total_mw = mb_full[2] - mb_full[0]
        cur_x = (w - total_mw) // 2 - mb_full[0]
        my = meta_y - mb_full[1]
        sep = "  ·  "
        sep_w = draw.textbbox((0, 0), sep, font=meta_font)[2] - draw.textbbox((0, 0), sep, font=meta_font)[0]
        sep_col = (254, 240, 138, 255) if is_tiffany else (251, 191, 36, 255)
        meta_text_col = (255, 255, 255, 255) if is_tiffany else (248, 250, 252, 255)
        for idx, part in enumerate(meta_parts):
            pb = draw.textbbox((0, 0), part, font=meta_font)
            pw = pb[2] - pb[0]
            draw.text((cur_x + 1, my + 1), part, font=meta_font, fill=(0, 0, 0, 190))
            draw.text((cur_x, my), part, font=meta_font, fill=meta_text_col)
            cur_x += pw
            if idx + 1 < len(meta_parts):
                draw.text((cur_x, my), sep, font=meta_font, fill=sep_col)
                cur_x += sep_w

    return out


def render_natural_cover_image(
    *,
    frame_img: Image.Image,
    faces: list[FaceBox],
    width: int,
    height: int,
    badge_text: str,
    dur_text: str,
    title: str,
    meta_parts: list[str],
) -> Image.Image:
    """Render natural-portrait cover with guest-locked crop (or 4:3 cinema fit-width for two-shots) + Tiffany card."""
    orig_w, orig_h = frame_img.size
    ar = width / max(1, height)
    s_ar = orig_w / max(1, orig_h)
    use_cinema_fit = False
    if 1.0 <= ar < 1.5 and orig_w > 0 and orig_h > 0 and s_ar > ar * 1.15 and len(faces) >= 2:
        base_vw = (orig_h * ar) / orig_w
        span_x = max(f.x + f.w for f in faces) - min(f.x for f in faces)
        if span_x > base_vw * 0.85:
            use_cinema_fit = True

    if use_cinema_fit:
        bg, safe_faces = compose_cinema_fit_width_frame(frame_img.convert("RGBA"), width, height, faces)
    else:
        safe_bottom_ratio = 0.68 if ar < 1.0 else 0.735
        bg, safe_faces = crop_frame_face_safe(frame_img.convert("RGBA"), width, height, faces, safe_bottom_ratio=safe_bottom_ratio)

    # Subtle top & bottom edge scrim so badges and Tiffany card read crisply without dulling bright areas
    scrim = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    s_draw = ImageDraw.Draw(scrim)
    grad_y0 = int(height * (0.64 if ar < 1.0 else 0.68))
    for y in range(grad_y0, height):
        alpha = int(105 * ((y - grad_y0) / max(1, height - grad_y0)) ** 1.35)
        s_draw.line([(0, y), (width, y)], fill=(5, 7, 12, alpha))
    top_y1 = int(height * 0.11)
    for y in range(0, top_y1):
        alpha = int(75 * (1.0 - y / max(1, top_y1)))
        s_draw.line([(0, y), (width, y)], fill=(5, 7, 12, alpha))
    bg = Image.alpha_composite(bg, scrim)

    _draw_top_badges(bg, badge_text=badge_text, dur_text=dur_text, faces=safe_faces, theme="tiffany")
    return _draw_frosted_glass_box(
        bg,
        title=title,
        meta_parts=meta_parts,
        faces=safe_faces,
        bottom_margin_ratio=0.032,
        min_width_ratio=0.64,
        max_width_ratio=0.92,
        theme="tiffany",
    )


def compose_pip_card_image(
    *,
    width: int,
    height: int,
    index_text: str,
    group_label: str,
    card_title: str,
    meta_parts: list[str],
    left_meta_lines: list[str],
    portrait_frame: SelectedFrame | None = None,
    enable_pip: bool = True,
) -> Image.Image:
    """
    Compose the dark charcoal + amber-gold editorial card, embedding a ~52.6% area
    16:9 portrait / two-shot Picture-in-Picture window on upper-right when available.
    """
    # Vertical charcoal gradient background + subtle warm amber aura behind PiP
    y_grad = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
    r_ch = (13.0 * (1.0 - y_grad) + 7.0 * y_grad)
    g_ch = (17.0 * (1.0 - y_grad) + 9.0 * y_grad)
    b_ch = (26.0 * (1.0 - y_grad) + 14.0 * y_grad)
    rgb = np.stack(
        [
            np.broadcast_to(r_ch, (height, width)),
            np.broadcast_to(g_ch, (height, width)),
            np.broadcast_to(b_ch, (height, width)),
        ],
        axis=2,
    ).astype(np.uint8)
    canvas = Image.fromarray(rgb, mode="RGB").convert("RGBA")

    has_pip = enable_pip and portrait_frame is not None and portrait_frame.image_path.exists()
    if has_pip and portrait_frame is not None:
        pip_x0 = int(width * 0.220)
        pip_y0 = int(height * 0.048)
        pip_x1 = int(width * 0.955)
        pip_y1 = int(height * 0.763)
        pip_w = pip_x1 - pip_x0
        pip_h = pip_y1 - pip_y0
        radius = max(12, int(height * 0.020))

        # Soft ambient shadow behind PiP frame
        shadow = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        sh_draw = ImageDraw.Draw(shadow)
        sh_draw.rounded_rectangle(
            [pip_x0 - 8, pip_y0 - 4, pip_x1 + 8, pip_y1 + 12],
            radius=radius + 4,
            fill=(0, 0, 0, 145),
        )
        shadow = shadow.filter(ImageFilter.GaussianBlur(radius=14))
        canvas = Image.alpha_composite(canvas, shadow)

        with Image.open(portrait_frame.image_path) as raw_im:
            pip_img = crop_pip_portrait(raw_im.convert("RGBA"), pip_w, pip_h, portrait_frame.faces)
        pip_mask = Image.new("L", (pip_w, pip_h), 0)
        ImageDraw.Draw(pip_mask).rounded_rectangle([0, 0, pip_w, pip_h], radius=radius, fill=255)
        canvas.paste(pip_img, (pip_x0, pip_y0), pip_mask)

        draw = ImageDraw.Draw(canvas)
        draw.rounded_rectangle(
            [pip_x0, pip_y0, pip_x1, pip_y1],
            radius=radius,
            outline=(245, 158, 11, 195),
            width=max(2, int(height * 0.0024)),
        )

        # Left editorial rail (x: 4.2% -> 20.0%)
        rail_x = int(width * 0.042)
        rail_w = int(width * 0.158)
        if index_text:
            idx_font = _load_cjk_font(max(36, int(height * 0.092)), weight="heavy")
            ib = draw.textbbox((0, 0), index_text, font=idx_font)
            draw.text((rail_x - ib[0], pip_y0 - ib[1]), index_text, font=idx_font, fill=(245, 158, 11, 255))
            rule_y = pip_y0 + (ib[3] - ib[1]) + int(height * 0.022)
        else:
            rule_y = pip_y0 + int(height * 0.015)
        draw.rectangle([rail_x, rule_y, rail_x + int(width * 0.036), rule_y + 5], fill=(251, 191, 36, 255))

        grp_font = _load_cjk_font(max(20, int(height * 0.030)), weight="bold")
        grp_lines = _wrap_text(draw, group_label or "高光精选", grp_font, rail_w)[:3]
        gy = rule_y + int(height * 0.024)
        for gl in grp_lines:
            draw.text((rail_x, gy), gl, font=grp_font, fill=(253, 230, 138, 255))
            gy += int(height * 0.042)

        sub_font = _load_cjk_font(max(15, int(height * 0.020)), weight="medium")
        sy = max(gy + int(height * 0.030), int(height * 0.50))
        for ln in left_meta_lines[:3]:
            if sy > int(height * 0.70):
                break
            for sub_ln in _wrap_text(draw, ln, sub_font, rail_w)[:2]:
                draw.text((rail_x, sy), sub_ln, font=sub_font, fill=(148, 163, 184, 235))
                sy += int(height * 0.030)

        return _draw_frosted_glass_box(
            canvas,
            title=card_title or group_label,
            meta_parts=meta_parts,
            faces=None,
            bottom_margin_ratio=0.030,
            min_width_ratio=0.88,
            max_width_ratio=0.915,
            bg_alpha=232,
            border_width=max(3, int(min(width, height) * 0.0038)),
            max_title_ratio=0.068,
            theme="obsidian",
        )

    # Fallback: classic dark charcoal + amber-gold editorial card without PiP
    draw = ImageDraw.Draw(canvas)
    margin_x = int(width * 0.12)
    draw.rectangle([margin_x, int(height * 0.30), margin_x + int(width * 0.055), int(height * 0.30) + 5], fill=(245, 158, 11))
    tag_font = _load_cjk_font(max(24, int(height * 0.032)), weight="bold")
    tag_str = f"{index_text}  /  {group_label or 'HIGHLIGHT'}" if index_text else (group_label or "HIGHLIGHT")
    draw.text((margin_x, int(height * 0.33)), tag_str, font=tag_font, fill=(251, 191, 36))
    return _draw_frosted_glass_box(
        canvas,
        title=card_title or group_label,
        meta_parts=meta_parts,
        faces=None,
        bottom_margin_ratio=0.14,
        min_width_ratio=0.76,
        max_width_ratio=0.84,
        bg_alpha=232,
        border_width=max(3, int(min(width, height) * 0.0038)),
        max_title_ratio=0.068,
        theme="obsidian",
    )


def _build_hero_pool_for_manifest(
    frame_src: Path,
    manifest: ProjectManifest,
    store: ProjectStore,
    cfg: dict[str, Any],
    prefer_two_shot: bool,
) -> list[SelectedFrame]:
    return build_hero_portrait_pool(
        frame_src,
        float(manifest.source.duration_s or 0.0),
        store.work_dir,
        prefer_two_shot=prefer_two_shot,
        guest_name=manifest.metadata.guest_zh or "",
        channel=manifest.metadata.channel_zh or manifest.metadata.channel or "",
        original_title=manifest.metadata.original_title or "",
        cfg=cfg,
    )


def _pick_portrait_frame(
    selected: SelectedFrame,
    hero_pool: list[SelectedFrame] | None = None,
) -> SelectedFrame:
    """
    For 3:4 portrait covers, keep clean solo Guest frames as-is, or upgrade two-shot frames
    to the highest-scoring solo Guest frame from hero_pool when available.
    """
    if (not selected.is_two_shot) and any(f.is_guest is True for f in selected.faces):
        return selected
    if hero_pool:
        solo_guests = [
            sf
            for sf in hero_pool
            if (not sf.is_two_shot)
            and any(f.is_guest is True for f in sf.faces)
            and sf.score >= 3.5
            and sf.image_path.exists()
        ]
        if solo_guests:
            solo_guests.sort(key=lambda x: x.score, reverse=True)
            return solo_guests[0]
    return selected


_PLATFORM_COVER_SPECS: dict[str, tuple[str, int, int, str]] = {
    "bili": (".cover.bili.jpg", 1920, 1080, "wide"),
    "bili_4x3": (".cover.bili.4x3.jpg", 1080, 810, "share"),
    "wechat": (".cover.wechat.jpg", 1080, 1440, "portrait"),
    "wechat_share": (".cover.wechat.share.jpg", 1080, 810, "share"),
}


_AUTOSUB_COVER_UTILS_CACHE: tuple[Path, float, Any] | None = None


def _resolve_autosub_cover_utils_path() -> Path | None:
    if _AUTOSUB_COVER_UTILS_PATH is not None:
        return _AUTOSUB_COVER_UTILS_PATH
    env_p = os.environ.get("CLIPCUT_AUTOSUB_COVER_UTILS", "").strip()
    if env_p:
        return Path(env_p).expanduser()
    local_p = str((load_local_config().get("cover") or {}).get("autosub_cover_utils") or "").strip()
    return Path(local_p).expanduser() if local_p else None


def _load_autosub_cover_utils() -> Any | None:
    global _AUTOSUB_COVER_UTILS_CACHE
    target = _resolve_autosub_cover_utils_path()
    if target is None or not target.is_file():
        return None
    try:
        mtime = float(target.stat().st_mtime)
        if _AUTOSUB_COVER_UTILS_CACHE is not None:
            cached_path, cached_mtime, cached_mod = _AUTOSUB_COVER_UTILS_CACHE
            if cached_path == target and cached_mtime == mtime:
                return cached_mod
        spec = importlib.util.spec_from_file_location("autosub_cover_utils", str(target))
        if spec is None or spec.loader is None:
            return None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _AUTOSUB_COVER_UTILS_CACHE = (target, mtime, mod)
        return mod
    except Exception:
        return None


def build_autosub_cover_comment(
    title: str,
    orientation: str,
    video_path: Path,
    published_date: str = "",
) -> bytes:
    """Build JPEG comment cache marker matching AutoSub cover_utils so uploaders reuse ClipCut covers."""
    cu = _load_autosub_cover_utils()
    if cu is not None:
        try:
            effective_date = cu._read_publish_date(str(video_path)) or published_date
            return cu._cover_cache_marker(title, orientation, effective_date)
        except Exception:
            pass
    metadata: dict[str, Any] = {
        "app": "autosub",
        "type": "cover",
        "style_version": "kaiti-alliance-lower-safe-v16",
        "orientation": orientation,
        "title": str(title),
        "title_sha256": hashlib.sha256(str(title).encode("utf-8")).hexdigest(),
    }
    if published_date:
        metadata["publish_date"] = published_date
    return json.dumps(metadata, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode("utf-8")


def _save_cover_jpeg(img: Image.Image, out_path: Path, comment: bytes | None = None) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_kwargs: dict[str, Any] = {"format": "JPEG", "quality": 93, "optimize": True}
    if comment is not None:
        save_kwargs["comment"] = comment
    img.convert("RGB").save(out_path, **save_kwargs)


def _render_cover_image_at_size(
    *,
    selected: SelectedFrame,
    width: int,
    height: int,
    badge_text: str,
    dur_text: str,
    title: str,
    meta_parts: list[str],
    style: str,
    index_text: str,
    group_label: str,
    left_meta_lines: list[str],
) -> Image.Image:
    if style == "editorial_pip" and width >= height:
        return compose_pip_card_image(
            width=width,
            height=height,
            index_text=index_text,
            group_label=group_label,
            card_title=title,
            meta_parts=meta_parts,
            left_meta_lines=left_meta_lines,
            portrait_frame=selected,
            enable_pip=True,
        )
    with Image.open(selected.image_path) as raw_im:
        return render_natural_cover_image(
            frame_img=raw_im,
            faces=selected.faces,
            width=width,
            height=height,
            badge_text=badge_text,
            dur_text=dur_text,
            title=title,
            meta_parts=meta_parts,
        )


def sync_existing_platform_covers(
    *,
    selected: SelectedFrame,
    cover_path: Path,
    badge_text: str,
    dur_text: str,
    title: str,
    meta_parts: list[str],
    style: str,
    index_text: str,
    group_label: str,
    left_meta_lines: list[str],
    published_date: str,
    hero_pool: list[SelectedFrame] | None = None,
) -> list[Path]:
    """Refresh any existing AutoSub platform cover siblings in place from clean raw portrait frames."""
    written: list[Path] = []
    video_path = cover_path.with_suffix(".mp4")
    for suffix, pw, ph, orient in _PLATFORM_COVER_SPECS.values():
        p = cover_path.with_name(f"{cover_path.stem}{suffix}")
        if not p.is_file():
            continue
        frame_choice = _pick_portrait_frame(selected, hero_pool) if pw < ph else selected
        img = _render_cover_image_at_size(
            selected=frame_choice,
            width=pw,
            height=ph,
            badge_text=badge_text,
            dur_text=dur_text,
            title=title,
            meta_parts=meta_parts,
            style=style,
            index_text=index_text,
            group_label=group_label,
            left_meta_lines=left_meta_lines,
        )
        _save_cover_jpeg(img, p, comment=build_autosub_cover_comment(title, orient, video_path, published_date))
        written.append(p)
    return written


def _resolve_clip_selected_frame(
    *,
    clean_video: Path,
    fallback_video: Path,
    clip: ClipItem,
    store: ProjectStore,
    hero_pool: list[SelectedFrame],
    fallback_index: int,
    prefer_two_shot: bool,
    reuse_cached_frame: bool,
) -> SelectedFrame | None:
    src_video = clean_video if clean_video.exists() else fallback_video
    guest_profile = load_guest_profile(store.work_dir)
    if reuse_cached_frame:
        cached_sel = store.work_dir / "cover_candidates" / f"{clip.clip_id}_selected.jpg"
        if cached_sel.exists():
            fmap = detect_faces_batch([cached_sel])
            raw_faces = fmap.get(str(cached_sel), [])
            sc, vfaces, two = score_candidate_frame(
                cached_sel,
                raw_faces,
                prefer_two_shot=prefer_two_shot,
                guest_profile=guest_profile,
                prefer_guest_solo=True,
            )
            has_guest_face = any(f.is_guest is True for f in vfaces)
            if vfaces and sc >= 1.2 and (guest_profile is None or has_guest_face):
                return SelectedFrame(
                    image_path=cached_sel,
                    timestamp_s=clip.start_s,
                    faces=vfaces,
                    is_two_shot=two,
                    score=sc,
                    from_fallback=False,
                )
        if hero_pool and clip.cover_file and Path(clip.cover_file).exists():
            guest_pool = [sf for sf in hero_pool if any(f.is_guest is True for f in sf.faces)] or hero_pool
            return guest_pool[fallback_index % len(guest_pool)]
    return select_best_frame_for_interval(
        source_video=src_video,
        start_s=clip.start_s,
        end_s=clip.end_s,
        work_dir=store.work_dir,
        cache_key=clip.clip_id,
        hero_pool=hero_pool,
        fallback_index=fallback_index,
        prefer_two_shot=prefer_two_shot,
        guest_profile=guest_profile,
        prefer_guest_solo=True,
    )


def _resolve_remix_selected_frame(
    *,
    manifest: ProjectManifest,
    store: ProjectStore,
    hero_pool: list[SelectedFrame],
    prefer_two_shot: bool,
) -> SelectedFrame | None:
    if hero_pool:
        return hero_pool[0]
    guest_profile = load_guest_profile(store.work_dir)
    cached_sel = store.work_dir / "cover_candidates" / "remix_main_cover_selected.jpg"
    if cached_sel.exists():
        fmap = detect_faces_batch([cached_sel])
        sc, vfaces, two = score_candidate_frame(
            cached_sel,
            fmap.get(str(cached_sel), []),
            prefer_two_shot=prefer_two_shot,
            guest_profile=guest_profile,
        )
        first_ts = manifest.remix.items[0].start_s if manifest.remix.items else 0.0
        if vfaces and sc >= 1.2 and (guest_profile is None or any(f.is_guest is True for f in vfaces)):
            return SelectedFrame(
                image_path=cached_sel,
                timestamp_s=first_ts,
                faces=vfaces,
                is_two_shot=two,
                score=sc,
                from_fallback=False,
            )
    if manifest.remix.items:
        clean_video = Path(manifest.source.analysis_video)
        src_video = clean_video if clean_video.exists() else Path(manifest.source.render_video)
        first = manifest.remix.items[0]
        return select_best_frame_for_interval(
            source_video=src_video,
            start_s=first.start_s,
            end_s=first.end_s,
            work_dir=store.work_dir,
            cache_key="remix_main_cover",
            hero_pool=hero_pool,
            fallback_index=0,
            prefer_two_shot=prefer_two_shot,
            guest_profile=guest_profile,
        )
    return None


def _compose_clip_cover_image(
    *,
    selected: SelectedFrame,
    clip: ClipItem,
    manifest: ProjectManifest,
    cfg: dict[str, Any],
    published_date: str,
    cover_path: Path,
    hero_pool: list[SelectedFrame] | None = None,
) -> bool:
    if not selected.image_path.exists():
        return False
    c_cfg = cfg.get("cover", {})
    style = str(c_cfg.get("style", "natural")).lower()
    dur = max(1.0, clip.end_s - clip.start_s)
    cats_zh = " · ".join(CATEGORY_LABELS_ZH.get(c, c) for c in clip.categories) or "精选切片"
    badge_text = f"#{clip.index:02d}  {cats_zh}"
    dur_text = _fmt_dur(dur)
    meta_parts = _build_cover_meta_parts(manifest, published_date)
    effective_title = (clip.cover_title or clip.title).strip()
    index_text = f"#{clip.index:02d}"
    left_meta = [f"时长 {dur_text}", manifest.metadata.channel_zh or manifest.metadata.channel or ""]

    w = manifest.source.width or 1920
    h = manifest.source.height or 1080
    out_img = _render_cover_image_at_size(
        selected=selected,
        width=w,
        height=h,
        badge_text=badge_text,
        dur_text=dur_text,
        title=effective_title,
        meta_parts=meta_parts,
        style=style,
        index_text=index_text,
        group_label=cats_zh,
        left_meta_lines=left_meta,
    )
    _save_cover_jpeg(out_img, cover_path)
    sync_existing_platform_covers(
        selected=selected,
        cover_path=cover_path,
        badge_text=badge_text,
        dur_text=dur_text,
        title=effective_title,
        meta_parts=meta_parts,
        style=style,
        index_text=index_text,
        group_label=cats_zh,
        left_meta_lines=left_meta,
        published_date=published_date,
        hero_pool=hero_pool,
    )
    return True


def _generate_clip_cover(
    *,
    clean_video: Path,
    fallback_video: Path,
    clip: ClipItem,
    manifest: ProjectManifest,
    store: ProjectStore,
    cfg: dict[str, Any],
    hero_pool: list[SelectedFrame],
    fallback_index: int,
    published_date: str,
    cover_path: Path,
    reuse_cached_frame: bool = False,
) -> bool:
    """Select an unsubtitled face-verified frame and render clip cover."""
    c_cfg = cfg.get("cover", {})
    prefer_two_shot = bool(c_cfg.get("prefer_two_shot", True))
    selected = _resolve_clip_selected_frame(
        clean_video=clean_video,
        fallback_video=fallback_video,
        clip=clip,
        store=store,
        hero_pool=hero_pool,
        fallback_index=fallback_index,
        prefer_two_shot=prefer_two_shot,
        reuse_cached_frame=reuse_cached_frame,
    )
    if selected is None:
        return False
    return _compose_clip_cover_image(
        selected=selected,
        clip=clip,
        manifest=manifest,
        cfg=cfg,
        published_date=published_date,
        cover_path=cover_path,
        hero_pool=hero_pool,
    )


def _generate_remix_cover(
    *,
    manifest: ProjectManifest,
    store: ProjectStore,
    cfg: dict[str, Any],
    hero_pool: list[SelectedFrame],
    published_date: str,
    cover_path: Path,
    total_duration_s: float | None = None,
) -> bool:
    """Render cover JPEG for Remix highlight video (supports natural & editorial_pip styles)."""
    c_cfg = cfg.get("cover", {})
    style = str(c_cfg.get("style", "natural")).lower()
    prefer_two_shot = bool(c_cfg.get("prefer_two_shot", True))
    picked = _resolve_remix_selected_frame(
        manifest=manifest,
        store=store,
        hero_pool=hero_pool,
        prefer_two_shot=prefer_two_shot,
    )
    if picked is None or not picked.image_path.exists():
        return False

    enabled_items = [it for it in sorted(manifest.remix.items, key=lambda x: x.order) if it.enabled]
    dur_s = total_duration_s if (total_duration_s is not None and total_duration_s > 0) else _estimate_remix_duration_s(manifest)

    dur_text = _fmt_dur(dur_s)
    meta_parts = _build_cover_meta_parts(manifest, published_date)
    w = manifest.source.width or 1920
    h = manifest.source.height or 1080
    remix_title = (manifest.remix.cover_title or manifest.remix.title or "全片高光集锦").strip()
    badge_text = "REMIX  ·  高光集锦"
    left_meta = [f"总长 {dur_text}", f"共 {len(enabled_items)} 段精华"]

    out_img = _render_cover_image_at_size(
        selected=picked,
        width=w,
        height=h,
        badge_text=badge_text,
        dur_text=dur_text,
        title=remix_title,
        meta_parts=meta_parts,
        style=style,
        index_text="REMIX",
        group_label="高光集锦",
        left_meta_lines=left_meta,
    )
    _save_cover_jpeg(out_img, cover_path)
    sync_existing_platform_covers(
        selected=picked,
        cover_path=cover_path,
        badge_text=badge_text,
        dur_text=dur_text,
        title=remix_title,
        meta_parts=meta_parts,
        style=style,
        index_text="REMIX",
        group_label="高光集锦",
        left_meta_lines=left_meta,
        published_date=published_date,
        hero_pool=hero_pool,
    )
    return True


def _generate_compilation_cover(
    *,
    manifest: ProjectManifest,
    store: ProjectStore,
    cfg: dict[str, Any],
    hero_pool: list[SelectedFrame],
    published_date: str,
    cover_path: Path,
    total_duration_s: float | None = None,
) -> bool:
    """Render cover JPEG for Compilation video."""
    c_cfg = cfg.get("cover", {})
    style = str(c_cfg.get("style", "natural")).lower()
    prefer_two_shot = bool(c_cfg.get("prefer_two_shot", True))
    picked = _resolve_remix_selected_frame(
        manifest=manifest,
        store=store,
        hero_pool=hero_pool,
        prefer_two_shot=prefer_two_shot,
    )
    if picked is None or not picked.image_path.exists():
        return False

    enabled_clips = [c for c in manifest.clips if c.enabled]
    dur_s = total_duration_s if (total_duration_s is not None and total_duration_s > 0) else sum(c.end_s - c.start_s for c in enabled_clips)
    dur_text = _fmt_dur(dur_s)
    meta_parts = _build_cover_meta_parts(manifest, published_date)
    comp_title = (manifest.compilation.cover_title or manifest.compilation.title or "切片精选集").strip()
    badge_text = "COMPILATION  ·  精选合辑"
    left_meta = [f"总长 {dur_text}", f"精选 {len(enabled_clips)} 支切片"]

    out_img = _render_cover_image_at_size(
        selected=picked,
        clean_img_path=picked.image_path,
        badge_text=badge_text,
        title_text=comp_title,
        meta_parts=meta_parts,
        left_meta_lines=left_meta,
        target_size=(1920, 1080),
        cfg=cfg,
        is_remix=True,
    )
    if out_img:
        _save_cover_jpeg(out_img, cover_path)
        sync_existing_platform_covers(
            selected=picked,
            cover_path=cover_path,
            badge_text=badge_text,
            dur_text=dur_text,
            title=comp_title,
            meta_parts=meta_parts,
            style=style,
            index_text="合辑",
            group_label="精选合辑",
            left_meta_lines=left_meta,
            published_date=published_date,
            hero_pool=hero_pool,
        )
        return True
    return False



def sync_platform_covers_for_target(
    manifest: ProjectManifest,
    store: ProjectStore,
    cfg: dict[str, Any],
    target: str,
    platforms: list[str],
    cover_title_override: str | None = None,
) -> list[Path]:
    """Pre-render platform-specific covers (Bilibili/WeChat/Xiaohongshu) in ClipCut's design with AutoSub cache markers."""
    published_date = _ensure_published_date(manifest, store)
    c_cfg = cfg.get("cover", {})
    style = str(c_cfg.get("style", "natural")).lower()
    prefer_two_shot = bool(c_cfg.get("prefer_two_shot", True))
    src_video = Path(manifest.source.render_video)
    clean_video = Path(manifest.source.analysis_video)
    frame_src = clean_video if clean_video.exists() else src_video
    hero_pool = _build_hero_pool_for_manifest(frame_src, manifest, store, cfg, prefer_two_shot)
    output_dir = Path(manifest.source.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    meta_parts = _build_cover_meta_parts(manifest, published_date)
    w = manifest.source.width or 1920
    h = manifest.source.height or 1080

    selected: SelectedFrame | None = None
    target_mp4: Path | None = None
    stored_title = ""
    badge_text = ""
    dur_text = ""
    index_text = ""
    group_label = ""
    left_meta_lines: list[str] = []

    if target == "remix":
        enabled_items = [it for it in sorted(manifest.remix.items, key=lambda x: x.order) if it.enabled]
        if not enabled_items:
            return []
        target_mp4 = Path(manifest.remix.output_file) if manifest.remix.output_file else (
            output_dir / build_remix_filename(manifest, published_date)
        )
        selected = _resolve_remix_selected_frame(
            manifest=manifest,
            store=store,
            hero_pool=hero_pool,
            prefer_two_shot=prefer_two_shot,
        )
        stored_title = (manifest.remix.cover_title or manifest.remix.title or "全片高光集锦").strip()
        dur_text = _fmt_dur(_estimate_remix_duration_s(manifest))
        badge_text = "REMIX  ·  高光集锦"
        index_text = "REMIX"
        group_label = "高光集锦"
        left_meta_lines = [f"总长 {dur_text}", f"共 {len(enabled_items)} 段精华"]
    elif target == "compilation":
        enabled_clips = [c for c in manifest.clips if c.enabled]
        if not enabled_clips:
            return []
        est_dur_s = sum(max(0.0, c.end_s - c.start_s) for c in enabled_clips) + len(enabled_clips) * float(
            manifest.compilation.card_duration_s or 1.8
        )
        target_mp4 = (
            Path(manifest.compilation.output_file)
            if manifest.compilation.output_file
            else (output_dir / build_compilation_filename(manifest, published_date, duration_s=est_dur_s))
        )
        selected = _resolve_remix_selected_frame(
            manifest=manifest,
            store=store,
            hero_pool=hero_pool,
            prefer_two_shot=prefer_two_shot,
        )
        stored_title = (manifest.compilation.cover_title or manifest.compilation.title or "切片精选集").strip()
        dur_text = _fmt_dur(est_dur_s)
        badge_text = "COMPILATION  ·  精选合辑"
        index_text = "合辑"
        group_label = "精选合辑"
        left_meta_lines = [f"总长 {dur_text}", f"精选 {len(enabled_clips)} 支切片"]
    else:
        for pos, clip in enumerate(manifest.clips):
            if clip.clip_id != target:
                continue
            target_mp4 = Path(clip.output_file) if clip.output_file else (
                output_dir / build_clip_filename(clip, manifest, published_date)
            )
            selected = _resolve_clip_selected_frame(
                clean_video=frame_src,
                fallback_video=target_mp4 if target_mp4.exists() else src_video,
                clip=clip,
                store=store,
                hero_pool=hero_pool,
                fallback_index=pos,
                prefer_two_shot=prefer_two_shot,
                reuse_cached_frame=True,
            )
            stored_title = (clip.cover_title or clip.title).strip()
            dur_text = _fmt_dur(max(1.0, clip.end_s - clip.start_s))
            cats_zh = " · ".join(CATEGORY_LABELS_ZH.get(c, c) for c in clip.categories) or "精选切片"
            badge_text = f"#{clip.index:02d}  {cats_zh}"
            index_text = f"#{clip.index:02d}"
            group_label = cats_zh
            left_meta_lines = [f"时长 {dur_text}", manifest.metadata.channel_zh or manifest.metadata.channel or ""]
            break

    if target_mp4 is None or selected is None or not selected.image_path.exists():
        return []

    effective_title = (cover_title_override or "").strip() or stored_title
    cover_jpg = target_mp4.with_suffix(".jpg")
    written: list[Path] = []
    seen_specs: set[str] = set()

    spec_keys: list[str] = []
    for plat in platforms:
        if plat == "bilibili":
            spec_keys.extend(["bili", "bili_4x3"])
        elif plat == "wechat":
            spec_keys.extend(["wechat", "wechat_share"])

    for key in spec_keys:
        if key in seen_specs or key not in _PLATFORM_COVER_SPECS:
            continue
        seen_specs.add(key)
        suffix, pw, ph, orient = _PLATFORM_COVER_SPECS[key]
        out_path = cover_jpg.with_name(f"{cover_jpg.stem}{suffix}")
        frame_choice = _pick_portrait_frame(selected, hero_pool) if pw < ph else selected
        img = _render_cover_image_at_size(
            selected=frame_choice,
            width=pw,
            height=ph,
            badge_text=badge_text,
            dur_text=dur_text,
            title=effective_title,
            meta_parts=meta_parts,
            style=style,
            index_text=index_text,
            group_label=group_label,
            left_meta_lines=left_meta_lines,
        )
        _save_cover_jpeg(img, out_path, comment=build_autosub_cover_comment(effective_title, orient, target_mp4, published_date))
        written.append(out_path)

    if "xiaohongshu" in platforms and (not cover_jpg.exists() or effective_title != stored_title):
        img = _render_cover_image_at_size(
            selected=selected,
            width=w,
            height=h,
            badge_text=badge_text,
            dur_text=dur_text,
            title=effective_title,
            meta_parts=meta_parts,
            style=style,
            index_text=index_text,
            group_label=group_label,
            left_meta_lines=left_meta_lines,
        )
        _save_cover_jpeg(img, cover_jpg)
        written.append(cover_jpg)

    return written


def refresh_single_cover(
    manifest: ProjectManifest,
    store: ProjectStore,
    cfg: dict[str, Any],
    target: str,
) -> Path | None:
    """Fast-refresh cover image for a single clip_id or 'remix' using cached portrait frames."""
    published_date = _ensure_published_date(manifest, store)
    c_cfg = cfg.get("cover", {})
    prefer_two_shot = bool(c_cfg.get("prefer_two_shot", True))
    src_video = Path(manifest.source.render_video)
    clean_video = Path(manifest.source.analysis_video)
    frame_src = clean_video if clean_video.exists() else src_video
    hero_pool = _build_hero_pool_for_manifest(frame_src, manifest, store, cfg, prefer_two_shot)
    output_dir = Path(manifest.source.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if target == "remix":
        enabled_items = [it for it in manifest.remix.items if it.enabled]
        if not enabled_items:
            return None
        target_mp4 = Path(manifest.remix.output_file) if manifest.remix.output_file else (
            output_dir / build_remix_filename(manifest, published_date)
        )
        cover_jpg = target_mp4.with_suffix(".jpg")
        effective_title = (manifest.remix.cover_title or manifest.remix.title or "全片高光集锦").strip()
        exp_hash = compute_cover_hash(
            key="remix",
            title=effective_title,
            categories=["remix"],
            start_s=enabled_items[0].start_s,
            end_s=enabled_items[-1].end_s,
            manifest=manifest,
            cfg=cfg,
            published_date=published_date,
        )
        ok = _generate_remix_cover(
            manifest=manifest,
            store=store,
            cfg=cfg,
            hero_pool=hero_pool,
            published_date=published_date,
            cover_path=cover_jpg,
        )
        if ok:
            manifest.remix.cover_file = str(cover_jpg)
            manifest.remix.cover_hash = exp_hash
            store.save(manifest)
            return cover_jpg
        return None

    if target == "compilation":
        enabled_clips = [c for c in manifest.clips if c.enabled]
        if not enabled_clips:
            return None
        est_dur_s = sum(max(0.0, c.end_s - c.start_s) for c in enabled_clips) + len(enabled_clips) * float(
            manifest.compilation.card_duration_s or 1.8
        )
        target_mp4 = (
            Path(manifest.compilation.output_file)
            if manifest.compilation.output_file
            else (output_dir / build_compilation_filename(manifest, published_date, duration_s=est_dur_s))
        )
        cover_jpg = target_mp4.with_suffix(".jpg")
        effective_title = (manifest.compilation.cover_title or manifest.compilation.title or "切片精选集").strip()
        exp_hash = compute_cover_hash(
            key="compilation",
            title=effective_title,
            categories=["compilation"],
            start_s=enabled_clips[0].start_s,
            end_s=enabled_clips[-1].end_s,
            manifest=manifest,
            cfg=cfg,
            published_date=published_date,
        )
        ok = _generate_compilation_cover(
            manifest=manifest,
            store=store,
            cfg=cfg,
            hero_pool=hero_pool,
            published_date=published_date,
            cover_path=cover_jpg,
            total_duration_s=est_dur_s,
        )
        if ok:
            manifest.compilation.cover_file = str(cover_jpg)
            manifest.compilation.cover_hash = exp_hash
            store.save(manifest)
            return cover_jpg
        return None

    for pos, clip in enumerate(manifest.clips):
        if clip.clip_id != target:
            continue
        target_mp4 = Path(clip.output_file) if clip.output_file else (
            output_dir / build_clip_filename(clip, manifest, published_date)
        )
        cover_jpg = target_mp4.with_suffix(".jpg")
        effective_title = (clip.cover_title or clip.title).strip()
        exp_hash = compute_cover_hash(
            key=clip.clip_id,
            title=effective_title,
            categories=clip.categories,
            start_s=clip.start_s,
            end_s=clip.end_s,
            manifest=manifest,
            cfg=cfg,
            published_date=published_date,
        )
        ok = _generate_clip_cover(
            clean_video=frame_src,
            fallback_video=target_mp4 if target_mp4.exists() else src_video,
            clip=clip,
            manifest=manifest,
            store=store,
            cfg=cfg,
            hero_pool=hero_pool,
            fallback_index=pos,
            published_date=published_date,
            cover_path=cover_jpg,
            reuse_cached_frame=True,
        )
        if ok:
            clip.cover_file = str(cover_jpg)
            clip.cover_hash = exp_hash
            store.save(manifest)
            return cover_jpg
        return None
    return None


def _create_title_card_video(
    *,
    width: int,
    height: int,
    fps: float,
    duration_s: float,
    fade_s: float,
    group_label: str,
    card_title: str,
    card_index: int = 0,
    out_mp4: Path,
    use_vt: bool,
    meta_parts: list[str] | None = None,
    left_meta_lines: list[str] | None = None,
    portrait_frame: SelectedFrame | None = None,
    enable_pip: bool = True,
    index_text: str = "",
) -> bool:
    """Render an editorial PiP title card PNG and encode to an MP4 segment matching main stream."""
    img = compose_pip_card_image(
        width=width,
        height=height,
        index_text=index_text,
        group_label=group_label or "高光精选",
        card_title=card_title or group_label,
        meta_parts=meta_parts or [],
        left_meta_lines=left_meta_lines or [],
        portrait_frame=portrait_frame,
        enable_pip=enable_pip,
    )

    png_path = out_mp4.with_suffix(".png")
    img.convert("RGB").save(png_path, format="PNG")

    fade_out_st = max(0.0, duration_s - fade_s)
    vf = f"fade=t=in:st=0:d={fade_s:.3f},fade=t=out:st={fade_out_st:.3f}:d={fade_s:.3f},fps={fps:.3f},format=yuv420p"

    base = [
        "ffmpeg",
        "-y",
        "-v",
        "error",
        "-loop",
        "1",
        "-t",
        f"{duration_s:.3f}",
        "-i",
        str(png_path),
        "-f",
        "lavfi",
        "-t",
        f"{duration_s:.3f}",
        "-i",
        "anullsrc=channel_layout=stereo:sample_rate=44100",
        "-vf",
        vf,
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-ar",
        "44100",
        "-ac",
        "2",
        "-shortest",
    ]
    cmd_vt = base + ["-c:v", "h264_videotoolbox", "-q:v", "60", str(out_mp4)]
    cmd_x264 = base + ["-c:v", "libx264", "-preset", "fast", "-crf", "18", str(out_mp4)]
    ok, _, _ = _run_ffmpeg_encode(cmd_vt, cmd_x264, use_vt)
    png_path.unlink(missing_ok=True)
    return ok


def _format_concat_file_entry(p: Path, base_dir: Path) -> str:
    """Format path for ffmpeg concat demuxer file list, escaping single quotes and using relative path if possible."""
    try:
        rel = p.resolve().relative_to(base_dir.resolve()).as_posix()
    except Exception:
        rel = p.resolve().as_posix()
    escaped = rel.replace("'", "'\\''")
    return f"file '{escaped}'"


def _render_remix(
    manifest: ProjectManifest,
    store: ProjectStore,
    cfg: dict[str, Any],
    use_vt: bool,
    run_id: str,
    hero_pool: list[SelectedFrame] | None = None,
    published_date: str = "",
) -> Path | None:
    remix = manifest.remix
    enabled_items = [it for it in sorted(remix.items, key=lambda x: x.order) if it.enabled]
    if not remix.enabled or not enabled_items:
        return None

    c_cfg = cfg.get("cover", {})
    prefer_two_shot = bool(c_cfg.get("prefer_two_shot", True))
    remix_card_pip = bool(c_cfg.get("remix_card_pip", True))
    if not published_date:
        published_date = _ensure_published_date(manifest, store)
    if hero_pool is None:
        clean_src = Path(manifest.source.analysis_video)
        hero_src = clean_src if clean_src.exists() else Path(manifest.source.render_video)
        hero_pool = _build_hero_pool_for_manifest(hero_src, manifest, store, cfg, prefer_two_shot)

    output_dir = Path(manifest.source.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    est_dur_s = _estimate_remix_duration_s(manifest)
    out_mp4 = output_dir / build_remix_filename(manifest, published_date, duration_s=est_dur_s)
    cover_jpg = out_mp4.with_suffix(".jpg")

    remix.render_hash = compute_remix_render_hash(manifest)
    effective_remix_title = (remix.cover_title or remix.title or "全片高光集锦").strip()
    exp_remix_cover_hash = compute_cover_hash(
        key="remix",
        title=effective_remix_title,
        categories=["remix"],
        start_s=enabled_items[0].start_s,
        end_s=enabled_items[-1].end_s,
        manifest=manifest,
        cfg=cfg,
        published_date=published_date,
    )

    if remix.render_hash == remix.last_rendered_hash:
        if not (out_mp4.exists() and out_mp4.stat().st_size > 1024):
            _migrate_cached_media(remix.output_file, out_mp4, cover_jpg)
        if out_mp4.exists() and out_mp4.stat().st_size > 1024:
            remix.output_file = str(out_mp4)
            if remix.cover_hash != exp_remix_cover_hash or not cover_jpg.exists():
                if _generate_remix_cover(
                    manifest=manifest,
                    store=store,
                    cfg=cfg,
                    hero_pool=hero_pool,
                    published_date=published_date,
                    cover_path=cover_jpg,
                ):
                    remix.cover_file = str(cover_jpg)
                    remix.cover_hash = exp_remix_cover_hash
            else:
                remix.cover_file = str(cover_jpg)
            store.append_receipt(
                run_id=run_id,
                stage="render",
                actor="system",
                action="render_remix_cache_hit",
                summary=f"Remix 命中增量缓存（hash {remix.render_hash}），跳过重渲: {out_mp4.name}",
            )
            return out_mp4

    t0 = time.monotonic()
    store.log(f"[render] 开始精装合成 Remix（{len(enabled_items)} 小段，卡片模式: {remix.card_mode}）")
    tmp_dir = store.work_dir / "remix_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    src_video = Path(manifest.source.render_video)
    clean_video = Path(manifest.source.analysis_video)
    frame_src = clean_video if clean_video.exists() else src_video
    w, h = manifest.source.width or 1920, manifest.source.height or 1080
    fps = manifest.source.fps if 15.0 <= manifest.source.fps <= 60.0 else 30.0
    defaults = ensure_default_assets()
    sfx_path = Path(remix.sfx_path) if (remix.sfx_path and Path(remix.sfx_path).exists()) else defaults["sfx"]
    bgm_path = Path(remix.bgm_path) if (remix.bgm_path and Path(remix.bgm_path).exists()) else defaults["bgm"]
    meta_parts = _build_cover_meta_parts(manifest, published_date)

    concat_files: list[Path] = []
    card_entry_times: list[float] = []
    cursor_s = 0.0
    last_group: str | None = None
    card_num = 0

    for idx, it in enumerate(enabled_items):
        group_key = remix_item_group_key(it)
        need_card = should_insert_title_card(remix.card_mode, idx, group_key, last_group)

        if need_card:
            card_num += 1
            card_mp4 = tmp_dir / f"card_{card_num:02d}.mp4"
            # Gather contiguous items in same group to pick best portrait frame in group window
            grp_start = it.start_s
            grp_end = it.end_s
            for nxt in enabled_items[idx + 1 :]:
                if remix_item_group_key(nxt) == group_key:
                    grp_end = max(grp_end, nxt.end_s)
                else:
                    break
            portrait_sf = (
                select_best_frame_for_interval(
                    source_video=frame_src,
                    start_s=grp_start,
                    end_s=grp_end,
                    work_dir=store.work_dir,
                    cache_key=f"rmx_{remix.render_hash}_{card_num:02d}",
                    hero_pool=hero_pool,
                    fallback_index=card_num - 1,
                    prefer_two_shot=prefer_two_shot,
                )
                if remix_card_pip
                else None
            )
            ok = _create_title_card_video(
                width=w,
                height=h,
                fps=fps,
                duration_s=remix.card_duration_s,
                fade_s=remix.card_fade_s,
                group_label=it.group_label or "高光片段",
                card_title=it.card_title or it.group_label,
                card_index=card_num,
                out_mp4=card_mp4,
                use_vt=use_vt,
                meta_parts=meta_parts,
                left_meta_lines=[
                    BEAT_LABELS_ZH.get(it.beat_type, "高光"),
                    published_date or (manifest.metadata.channel_zh or manifest.metadata.channel or ""),
                ],
                portrait_frame=portrait_sf,
                enable_pip=remix_card_pip,
                index_text="",
            )
            if ok:
                concat_files.append(card_mp4)
                card_entry_times.append(cursor_s)
                cursor_s += remix.card_duration_s
        last_group = group_key

        # Extract loudness-normalized excerpt matching exact stream geometry
        seg_mp4 = tmp_dir / f"seg_{idx + 1:02d}.mp4"
        dur = max(0.5, it.end_s - it.start_s)
        fade_s = min(0.03, dur * 0.15)
        fade_out_st = max(0.0, dur - fade_s)
        vf = f"scale={w}:{h}:force_original_aspect_ratio=decrease,pad={w}:{h}:(ow-iw)/2:(oh-ih)/2,fps={fps:.3f},format=yuv420p"
        af = f"loudnorm=I=-16:TP=-1.5:LRA=11,afade=t=in:st=0:d={fade_s:.3f},afade=t=out:st={fade_out_st:.3f}:d={fade_s:.3f}"

        base = [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-ss",
            f"{it.start_s:.3f}",
            "-i",
            str(src_video),
            "-t",
            f"{dur:.3f}",
            "-vf",
            vf,
            "-af",
            af,
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-ar",
            "44100",
            "-ac",
            "2",
        ]
        cmd_vt = base + ["-c:v", "h264_videotoolbox", "-q:v", "60", str(seg_mp4)]
        cmd_x264 = base + ["-c:v", "libx264", "-preset", "fast", "-crf", "18", str(seg_mp4)]
        ok, _, err = _run_ffmpeg_encode(cmd_vt, cmd_x264, use_vt)
        if not ok:
            store.log(f"[render] Remix 片段 {it.remix_id} 截取失败: {err}")
            continue
        concat_files.append(seg_mp4)
        cursor_s += dur

    if not concat_files:
        return None

    concat_list = tmp_dir / "concat.txt"
    concat_list.write_text(
        "\n".join(_format_concat_file_entry(p, tmp_dir) for p in concat_files) + "\n",
        encoding="utf-8",
    )
    raw_concat = tmp_dir / "remix_raw.mp4"
    res_cat = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_list),
            "-c",
            "copy",
            str(raw_concat),
        ],
        capture_output=True,
        text=True,
    )
    if res_cat.returncode != 0:
        # Fallback re-encode concat if stream params differed slightly
        res_fallback = subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-v",
                "error",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(concat_list),
                "-c:v",
                "libx264",
                "-preset",
                "fast",
                "-crf",
                "18",
                "-c:a",
                "aac",
                "-b:a",
                "192k",
                str(raw_concat),
            ],
            capture_output=True,
            text=True,
        )
        if res_fallback.returncode != 0:
            err_msg = (res_fallback.stderr or res_cat.stderr or "未知错误").strip()
            store.log(f"[render] Remix 拼接失败: {err_msg}")
            return None

    # Audio mastering pass: mix transition SFX + BGM with sidechain ducking
    r_cfg = cfg.get("remix", {})
    need_audio_mix = (remix.enable_bgm and bgm_path.exists()) or (remix.enable_sfx and sfx_path.exists() and card_entry_times)

    if not need_audio_mix:
        raw_concat.replace(out_mp4)
    else:
        inputs = ["-i", str(raw_concat)]
        filter_parts: list[str] = []
        mix_labels: list[str] = []

        # Split main voice track so sidechaincompress can listen to it while we keep clean voice in final mix
        if remix.enable_bgm and bgm_path.exists():
            filter_parts.append("[0:a]asplit=2[voice_main][voice_sc]")
            mix_labels.append("[voice_main]")
        else:
            mix_labels.append("[0:a]")

        next_idx = 1
        if remix.enable_bgm and bgm_path.exists():
            inputs += ["-stream_loop", "-1", "-i", str(bgm_path)]
            bgm_idx = next_idx
            next_idx += 1
            fade_out_st = max(0.0, cursor_s - 1.5)
            duck_th = float(r_cfg.get("duck_threshold", 0.03))
            duck_ratio = float(r_cfg.get("duck_ratio", 6.0))
            duck_atk = int(r_cfg.get("duck_attack_ms", 40))
            duck_rel = int(r_cfg.get("duck_release_ms", 350))
            filter_parts.append(
                f"[{bgm_idx}:a]atrim=0:{cursor_s:.3f},asetpts=PTS-STARTPTS,"
                f"volume={remix.bgm_volume:.3f},afade=t=in:st=0:d=0.6,afade=t=out:st={fade_out_st:.3f}:d=1.5[bgm_bed]"
            )
            filter_parts.append(
                f"[bgm_bed][voice_sc]sidechaincompress=threshold={duck_th}:ratio={duck_ratio}:attack={duck_atk}:release={duck_rel}[bgm_ducked]"
            )
            mix_labels.append("[bgm_ducked]")

        if remix.enable_sfx and sfx_path.exists() and card_entry_times:
            inputs += ["-i", str(sfx_path)]
            sfx_idx = next_idx
            next_idx += 1
            n_sfx = len(card_entry_times)
            if n_sfx == 1:
                delay_ms = max(0, int(card_entry_times[0] * 1000))
                filter_parts.append(
                    f"[{sfx_idx}:a]volume={remix.sfx_volume:.3f},adelay={delay_ms}|{delay_ms}[sfx_0]"
                )
                mix_labels.append("[sfx_0]")
            else:
                split_outs = "".join(f"[sfx_src_{i}]" for i in range(n_sfx))
                filter_parts.append(f"[{sfx_idx}:a]volume={remix.sfx_volume:.3f},asplit={n_sfx}{split_outs}")
                for i, t_card in enumerate(card_entry_times):
                    delay_ms = max(0, int(t_card * 1000))
                    filter_parts.append(f"[sfx_src_{i}]adelay={delay_ms}|{delay_ms}[sfx_{i}]")
                    mix_labels.append(f"[sfx_{i}]")

        filter_parts.append(
            f"{''.join(mix_labels)}amix=inputs={len(mix_labels)}:duration=first:dropout_transition=0:normalize=0,alimiter=limit=0.95[aout]"
        )
        fc = ";".join(filter_parts)

        res_mix = subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-v",
                "error",
                *inputs,
                "-filter_complex",
                fc,
                "-map",
                "0:v:0",
                "-map",
                "[aout]",
                "-c:v",
                "copy",
                "-c:a",
                "aac",
                "-b:a",
                "192k",
                "-movflags",
                "+faststart",
                str(out_mp4),
            ],
            capture_output=True,
            text=True,
        )
        if res_mix.returncode != 0:
            # Safe fallback: keep raw concat so Remix never fails completely
            store.log(f"[render] Remix 混音降级为无BGM/音效版本: {res_mix.stderr[-300:]}")
            raw_concat.replace(out_mp4)

    remix.last_rendered_hash = remix.render_hash
    remix.output_file = str(out_mp4)
    if _generate_remix_cover(
        manifest=manifest,
        store=store,
        cfg=cfg,
        hero_pool=hero_pool,
        published_date=published_date,
        cover_path=cover_jpg,
        total_duration_s=cursor_s,
    ):
        remix.cover_file = str(cover_jpg)
        remix.cover_hash = exp_remix_cover_hash
    elapsed_ms = int((time.monotonic() - t0) * 1000)
    store.append_receipt(
        run_id=run_id,
        stage="render",
        actor="system",
        action="render_remix",
        elapsed_ms=elapsed_ms,
        summary=f"Remix 精装合成完成: {out_mp4.name}（总长 {cursor_s:.1f}s，{len(enabled_items)} 段，标题卡 {card_num} 张，BGM={'开' if remix.enable_bgm else '关'}，音效={'开' if remix.enable_sfx else '关'}）",
        details={
            "output_file": str(out_mp4),
            "cover_file": remix.cover_file,
            "duration_s": round(cursor_s, 1),
            "cards": card_num,
            "render_hash": remix.render_hash,
        },
    )
    return out_mp4


def _render_compilation(
    manifest: ProjectManifest,
    store: ProjectStore,
    cfg: dict[str, Any],
    use_vt: bool,
    run_id: str,
    hero_pool: list[SelectedFrame] | None = None,
    published_date: str = "",
) -> Path | None:
    """Render Selected Clips Compilation (sequential concatenation of enabled clips with title cards)."""
    t0 = time.monotonic()
    comp = manifest.compilation
    enabled_clips = [c for c in manifest.clips if c.enabled]
    if not comp.enabled or not enabled_clips:
        return None

    c_cfg = cfg.get("cover", {})
    prefer_two_shot = bool(c_cfg.get("prefer_two_shot", True))
    if not published_date:
        published_date = _ensure_published_date(manifest, store)
    if hero_pool is None:
        clean_src = Path(manifest.source.analysis_video)
        hero_src = clean_src if clean_src.exists() else Path(manifest.source.render_video)
        hero_pool = _build_hero_pool_for_manifest(hero_src, manifest, store, cfg, prefer_two_shot)

    output_dir = Path(manifest.source.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    card_dur = float(comp.card_duration_s or 1.8)
    card_fade = float(comp.card_fade_s or 0.2)
    est_dur_s = sum(max(0.0, c.end_s - c.start_s) for c in enabled_clips) + len(enabled_clips) * card_dur
    out_mp4 = output_dir / build_compilation_filename(manifest, published_date, duration_s=est_dur_s)
    cover_jpg = out_mp4.with_suffix(".jpg")

    comp.render_hash = compute_compilation_render_hash(manifest)
    effective_title = (comp.cover_title or comp.title or "切片精选集").strip()
    exp_cover_hash = compute_cover_hash(
        key="compilation",
        title=effective_title,
        categories=["compilation"],
        start_s=enabled_clips[0].start_s,
        end_s=enabled_clips[-1].end_s,
        manifest=manifest,
        cfg=cfg,
        published_date=published_date,
    )

    if comp.render_hash == comp.last_rendered_hash:
        if not (out_mp4.exists() and out_mp4.stat().st_size > 1024):
            _migrate_cached_media(comp.output_file, out_mp4, cover_jpg)
        if out_mp4.exists() and out_mp4.stat().st_size > 1024:
            comp.output_file = str(out_mp4)
            if comp.cover_hash != exp_cover_hash or not cover_jpg.exists():
                if _generate_compilation_cover(
                    manifest=manifest,
                    store=store,
                    cfg=cfg,
                    hero_pool=hero_pool,
                    published_date=published_date,
                    cover_path=cover_jpg,
                    total_duration_s=est_dur_s,
                ):
                    comp.cover_file = str(cover_jpg)
                    comp.cover_hash = exp_cover_hash
            else:
                comp.cover_file = str(cover_jpg)
            store.save(manifest)
            return out_mp4

    tmp_dir = store.work_dir / "compilation_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    meta_parts = _build_cover_meta_parts(manifest, published_date)
    src_video = Path(manifest.source.render_video)
    w = manifest.source.width or 1920
    h = manifest.source.height or 1080
    fps = manifest.source.fps or 30.0

    concat_files: list[Path] = []
    store.log(f"[render] 开始合成切片精选集 ({len(enabled_clips)} 支已选切片，预估总长 {est_dur_s:.1f}s)")

    for idx, clip in enumerate(enabled_clips, 1):
        # 1. Chapter title card
        card_mp4 = tmp_dir / f"card_{idx:02d}.mp4"
        card_title = (clip.cover_title or clip.title).strip()
        part_tag = (
            CATEGORY_LABELS_ZH.get(clip.categories[0], clip.categories[0])
            if clip.categories
            else (clip.chapter_title or "切片精选")
        )
        portrait = _pick_portrait_frame(hero_pool, clip.start_s, clip.end_s)
        _create_title_card_video(
            width=w,
            height=h,
            fps=fps,
            duration_s=card_dur,
            fade_s=card_fade,
            group_label=part_tag,
            card_title=card_title,
            card_index=idx,
            out_mp4=card_mp4,
            use_vt=use_vt,
            meta_parts=meta_parts,
            portrait_frame=portrait,
            enable_pip=True,
            index_text="",
        )
        concat_files.append(card_mp4)

        # 2. Clip segment
        clip_mp4 = Path(clip.output_file) if (clip.output_file and Path(clip.output_file).exists() and Path(clip.output_file).stat().st_size > 1024) else None
        if clip_mp4:
            concat_files.append(clip_mp4)
        else:
            seg_mp4 = tmp_dir / f"seg_{idx:02d}.mp4"
            fade_ms = int(cfg.get("boundary", {}).get("audio_fade_ms", 25))
            ok, enc, err = _render_single_clip(src_video, clip, seg_mp4, fade_ms, use_vt)
            if ok and seg_mp4.exists():
                concat_files.append(seg_mp4)
            else:
                store.log(f"[render] 精选集分段 {clip.clip_id} 截取失败: {err}")

    if not concat_files:
        store.log("[render] 精选集无有效拼接分段")
        return None

    concat_txt = tmp_dir / "concat.txt"
    concat_txt.write_text(
        "\n".join(_format_concat_file_entry(p, tmp_dir) for p in concat_files) + "\n",
        encoding="utf-8",
    )
    raw_mp4 = tmp_dir / "compilation_raw.mp4"
    res_cat = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_txt),
            "-c",
            "copy",
            "-movflags",
            "+faststart",
            str(raw_mp4),
        ],
        capture_output=True,
        text=True,
    )
    if res_cat.returncode != 0:
        res_fb = subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-v",
                "error",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(concat_txt),
                "-c:v",
                "libx264",
                "-preset",
                "fast",
                "-crf",
                "18",
                "-c:a",
                "aac",
                "-b:a",
                "192k",
                "-movflags",
                "+faststart",
                str(raw_mp4),
            ],
            capture_output=True,
            text=True,
        )
        if res_fb.returncode != 0:
            store.log(f"[render] 精选集 ffmpeg concat 失败: {res_fb.stderr}")
            return None

    if raw_mp4.exists() and raw_mp4.stat().st_size > 1024:
        raw_mp4.replace(out_mp4)
        comp.output_file = str(out_mp4)
        comp.last_rendered_hash = comp.render_hash
        _generate_compilation_cover(
            manifest=manifest,
            store=store,
            cfg=cfg,
            hero_pool=hero_pool,
            published_date=published_date,
            cover_path=cover_jpg,
            total_duration_s=est_dur_s,
        )
        comp.cover_file = str(cover_jpg)
        comp.cover_hash = exp_cover_hash
        elapsed_ms = int((time.monotonic() - t0) * 1000)
        store.append_receipt(
            run_id=run_id,
            stage="render",
            actor="system",
            action="render_compilation",
            elapsed_ms=elapsed_ms,
            summary=f"切片精选集完成: {out_mp4.name}（包含 {len(enabled_clips)} 支已选切片，插入片头卡片 {len(enabled_clips)} 张，总长 {est_dur_s:.1f}s）",
            details={
                "output_file": str(out_mp4),
                "cover_file": comp.cover_file,
                "duration_s": round(est_dur_s, 1),
                "clips_count": len(enabled_clips),
                "render_hash": comp.render_hash,
            },
        )
        store.save(manifest)
        store.log(f"[render] 切片精选集渲染完成: {out_mp4.name}")
        return out_mp4
    return None



def _write_markdown_deliverables(manifest: ProjectManifest, output_dir: Path) -> None:
    date_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    title = manifest.metadata.original_theme_zh or manifest.metadata.original_title
    enabled_clips = [c for c in manifest.clips if c.enabled]

    # 1. clips_index.md
    idx_lines = [
        f"# 《{title}》切片总览",
        "",
        f"> 生成时间：{date_str} | 切片数：{len(enabled_clips)} | 主话题：{', '.join(manifest.primary_topics) or '—'}",
        "",
        "| # | 标题 | 类别 | 时长 | 时间段 | 文件 |",
        "|---|------|------|------|--------|------|",
    ]
    for c in enabled_clips:
        cats = " / ".join(CATEGORY_LABELS_ZH.get(x, x) for x in c.categories)
        fname = Path(c.output_file).name if c.output_file else "未渲染"
        idx_lines.append(
            f"| {c.index:02d} | {c.title} | {cats} | {_fmt_dur(c.end_s - c.start_s)} | "
            f"`{_fmt_dur(c.start_s)} → {_fmt_dur(c.end_s)}` | `{fname}` |"
        )
    if manifest.remix.output_file:
        idx_lines.append(
            f"| 00 | {manifest.remix.title} | 混剪集锦 | — | {len([x for x in manifest.remix.items if x.enabled])} 段 | `{Path(manifest.remix.output_file).name}` |"
        )
    (output_dir / "clips_index.md").write_text("\n".join(idx_lines) + "\n", encoding="utf-8")

    # 2. clips_detail.md
    det_lines = [
        f"# 《{title}》切片详情与编审记录",
        "",
        f"> 原标题：{manifest.metadata.original_title}",
        f"> 嘉宾：{manifest.metadata.guest_zh or '—'} | 频道：{manifest.metadata.channel_zh or manifest.metadata.channel or '—'}",
        f"> 主话题去重集合：{', '.join(manifest.primary_topics) or '—'}",
        "",
        "---",
        "",
    ]
    for c in enabled_clips:
        cats = " / ".join(CATEGORY_LABELS_ZH.get(x, x) for x in c.categories)
        hot = f"（命中热点：{', '.join(c.matched_hot_topics)}）" if c.matched_hot_topics else ""
        det_lines += [
            f"## {c.index:02d} · {c.title}",
            "",
            f"- 类别：{cats}{hot}",
            f"- 时间段：`{_fmt_dur(c.start_s)}` → `{_fmt_dur(c.end_s)}`（时长 {_fmt_dur(c.end_s - c.start_s)}，字幕句 `u{c.start_utt:04d}`–`u{c.end_utt:04d}`，边界吸附 `{c.snap_delta_start_ms:+d}ms / {c.snap_delta_end_ms:+d}ms`）",
            f"- 入选理由：{c.reason}",
            f"- 开头原话：{c.hook}",
            f"- 简介：{c.summary}",
        ]
        if c.key_quote:
            det_lines.append(f"- 核心金句：\"{c.key_quote}\"")
        if c.notes:
            det_lines.append("- 编审说明：" + "；".join(c.notes))
        if c.output_file:
            det_lines.append(f"- 输出文件：`{Path(c.output_file).name}`")
        det_lines += ["", "---", ""]

    # Honest delivery: dropped topics & suppressed hot topics
    dropped = [t for t in manifest.topic_map if not t.selected_clip_id]
    if dropped:
        det_lines += ["## 未切话题清单（诚实交付）", ""]
        for t in dropped:
            det_lines.append(
                f"- `{_fmt_dur(t.start_s)}–{_fmt_dur(t.end_s)}` **{t.title}**：{t.dropped_reason or t.filler_reason or '未达阈值'}"
            )
        det_lines.append("")

    if manifest.suppressed_hot_topics:
        det_lines += ["## 主话题去重拦截记录", ""]
        for s in manifest.suppressed_hot_topics:
            det_lines.append(f"- 关键词 `{s.matched_keyword}`（冲突主话题 `{s.primary_topic_conflict}`）：{s.reason}")
        det_lines.append("")

    (output_dir / "clips_detail.md").write_text("\n".join(det_lines), encoding="utf-8")

    # 3. remix_script.md
    enabled_remix = [it for it in sorted(manifest.remix.items, key=lambda x: x.order) if it.enabled]
    speech_total_s = sum(max(0.0, it.end_s - it.start_s) for it in enabled_remix)
    est_cards = 0
    last_gk: str | None = None
    for idx, it in enumerate(enabled_remix):
        gk = remix_item_group_key(it)
        if should_insert_title_card(manifest.remix.card_mode, idx, gk, last_gk):
            est_cards += 1
        last_gk = gk
    est_total_s = speech_total_s + est_cards * (manifest.remix.card_duration_s if manifest.remix.card_mode != "none" else 0.0)
    src_dur = max(1.0, float(manifest.source.duration_s or 1.0))
    ratio_pct = (est_total_s / src_dur) * 100.0 if manifest.source.duration_s else 0.0

    rmx_lines = [
        f"# {manifest.remix.title}",
        "",
        f"> 卡片模式：`{manifest.remix.card_mode}`（{est_cards} 张卡） | 板块：{len({remix_item_group_key(x) for x in enabled_remix})} 组 | 段落：{len(enabled_remix)} 段 | 预估总长：{_fmt_dur(est_total_s)}（人声 {_fmt_dur(speech_total_s)}，占原片 {ratio_pct:.1f}%） | BGM：`{'开' if manifest.remix.enable_bgm else '关'}` | 音效：`{'开' if manifest.remix.enable_sfx else '关'}`",
    ]
    if manifest.remix.editorial_thesis:
        rmx_lines += ["", f"> 总编策展思路：{manifest.remix.editorial_thesis}"]
    rmx_lines += [
        "",
        "| 顺序 | 节拍 | 分组 | 卡片标题 | 时间段 | 原话与叙事说明 |",
        "|------|------|------|----------|--------|----------------|",
    ]
    for it in enabled_remix:
        beat_zh = BEAT_LABELS_ZH.get(it.beat_type, it.role)
        quote_clean = it.quote.replace("|", "\\|").replace("\n", " ")[:140]
        note_clean = it.reason.replace("|", "\\|").replace("\n", " ")
        detail_col = f"{quote_clean}" + (f"<br><small>说明：{note_clean}</small>" if note_clean else "")
        rmx_lines.append(
            f"| {it.order:02d} | {beat_zh} | {it.group_label} | {it.card_title} | "
            f"`{_fmt_dur(it.start_s)}→{_fmt_dur(it.end_s)}` ({it.end_s - it.start_s:.1f}s) | {detail_col} |"
        )
    (output_dir / "remix_script.md").write_text("\n".join(rmx_lines) + "\n", encoding="utf-8")

    # 4. BaoCut handoff JSON plan
    baocut_plan = {
        "format": "clipcut-baocut-handoff/1",
        "source_video": manifest.source.render_video,
        "subtitle_file": manifest.source.subtitle_path,
        "clips": [
            {
                "name": c.title,
                "start": c.start_s,
                "end": c.end_s,
                "hook": c.hook,
                "reason": c.reason,
                "notes": c.notes[:8],
            }
            for c in enabled_clips
        ],
        "remix": [
            {
                "order": it.order,
                "group": it.group_label,
                "card_title": it.card_title,
                "start": it.start_s,
                "end": it.end_s,
            }
            for it in sorted(manifest.remix.items, key=lambda x: x.order)
            if it.enabled
        ],
    }
    (output_dir / "baocut_export_plan.json").write_text(
        json.dumps(baocut_plan, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def run_render_stage(
    manifest: ProjectManifest,
    store: ProjectStore,
    cfg: dict[str, Any],
    run_id: str,
    *,
    force: bool = False,
) -> ProjectManifest:
    t0 = time.monotonic()
    src_video = Path(manifest.source.render_video)
    clean_video = Path(manifest.source.analysis_video)
    frame_src = clean_video if clean_video.exists() else src_video
    output_dir = Path(manifest.source.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fade_ms = int(cfg.get("boundary", {}).get("audio_fade_ms", 25))
    use_vt = _has_videotoolbox()
    published_date = _ensure_published_date(manifest, store)
    prefer_two_shot = bool(cfg.get("cover", {}).get("prefer_two_shot", True))
    hero_pool = _build_hero_pool_for_manifest(frame_src, manifest, store, cfg, prefer_two_shot)

    rendered_count = 0
    cached_count = 0
    cover_refreshed = 0
    failed_count = 0

    for clip_pos, clip in enumerate(manifest.clips):
        if not clip.enabled:
            continue
        clip.render_hash = compute_clip_render_hash(clip, manifest.source.render_video)
        fname = build_clip_filename(clip, manifest, published_date)
        out_mp4 = output_dir / fname
        cover_jpg = out_mp4.with_suffix(".jpg")
        effective_cover_title = (clip.cover_title or clip.title).strip()
        exp_cover_hash = compute_cover_hash(
            key=clip.clip_id,
            title=effective_cover_title,
            categories=clip.categories,
            start_s=clip.start_s,
            end_s=clip.end_s,
            manifest=manifest,
            cfg=cfg,
            published_date=published_date,
        )

        if not force and clip.render_hash == clip.last_rendered_hash:
            if not (out_mp4.exists() and out_mp4.stat().st_size > 1024):
                _migrate_cached_media(clip.output_file, out_mp4, cover_jpg)
            if out_mp4.exists() and out_mp4.stat().st_size > 1024:
                clip.output_file = str(out_mp4)
                if clip.cover_hash != exp_cover_hash or not cover_jpg.exists():
                    if _generate_clip_cover(
                        clean_video=frame_src,
                        fallback_video=out_mp4,
                        clip=clip,
                        manifest=manifest,
                        store=store,
                        cfg=cfg,
                        hero_pool=hero_pool,
                        fallback_index=clip_pos,
                        published_date=published_date,
                        cover_path=cover_jpg,
                        reuse_cached_frame=True,
                    ):
                        clip.cover_file = str(cover_jpg)
                        clip.cover_hash = exp_cover_hash
                        cover_refreshed += 1
                else:
                    clip.cover_file = str(cover_jpg)
                cached_count += 1
                continue

        store.log(f"[render] 帧精准切片 {clip.clip_id}: {fname} ({clip.start_s:.2f}s → {clip.end_s:.2f}s)")
        ct0 = time.monotonic()
        ok, encoder, err = _render_single_clip(src_video, clip, out_mp4, fade_ms, use_vt)
        celapsed = int((time.monotonic() - ct0) * 1000)

        if ok:
            clip.last_rendered_hash = clip.render_hash
            clip.output_file = str(out_mp4)
            if _generate_clip_cover(
                clean_video=frame_src,
                fallback_video=out_mp4,
                clip=clip,
                manifest=manifest,
                store=store,
                cfg=cfg,
                hero_pool=hero_pool,
                fallback_index=clip_pos,
                published_date=published_date,
                cover_path=cover_jpg,
            ):
                clip.cover_file = str(cover_jpg)
                clip.cover_hash = exp_cover_hash
            rendered_count += 1
            store.append_receipt(
                run_id=run_id,
                stage="render",
                actor="system",
                action="render_clip",
                elapsed_ms=celapsed,
                summary=f"切出 {fname}（{clip.end_s - clip.start_s:.1f}s，编码器: {encoder}）",
                details={
                    "clip_id": clip.clip_id,
                    "encoder": encoder,
                    "output_file": str(out_mp4),
                    "cover_file": clip.cover_file,
                    "render_hash": clip.render_hash,
                },
            )
        else:
            failed_count += 1
            store.append_receipt(
                run_id=run_id,
                stage="render",
                actor="system",
                action="render_clip",
                elapsed_ms=celapsed,
                summary=f"切片 {clip.clip_id} 渲染失败",
                error=err,
            )

    if manifest.remix.enabled and any(x.enabled for x in manifest.remix.items):
        if force:
            manifest.remix.last_rendered_hash = ""
        _render_remix(
            manifest,
            store,
            cfg,
            use_vt,
            run_id,
            hero_pool=hero_pool,
            published_date=published_date,
        )

    if manifest.compilation.enabled and any(c.enabled for c in manifest.clips):
        if force:
            manifest.compilation.last_rendered_hash = ""
        _render_compilation(
            manifest,
            store,
            cfg,
            use_vt,
            run_id,
            hero_pool=hero_pool,
            published_date=published_date,
        )
    _write_markdown_deliverables(manifest, output_dir)

    store.save(manifest)
    total_ms = int((time.monotonic() - t0) * 1000)
    store.append_receipt(
        run_id=run_id,
        stage="render",
        actor="system",
        action="render_complete",
        elapsed_ms=total_ms,
        summary=f"渲染完成：新切 {rendered_count} 支，命中缓存 {cached_count} 支，刷新封面 {cover_refreshed} 张，失败 {failed_count} 支；输出目录 {output_dir}",
        details={
            "rendered": rendered_count,
            "cached": cached_count,
            "covers_refreshed": cover_refreshed,
            "failed": failed_count,
            "output_dir": str(output_dir),
            "videotoolbox": use_vt,
        },
    )
    return manifest


def render_single_clip_and_cover(
    manifest: ProjectManifest,
    store: ProjectStore,
    clip_id: str,
    cfg: dict[str, Any] | None = None,
    *,
    force: bool = False,
) -> tuple[bool, str]:
    """定向渲染单个切片及其封面，更新 manifest 并返回结果"""
    if cfg is None:
        cfg = load_editorial_config()
    target_pos = next((i for i, c in enumerate(manifest.clips) if c.clip_id == clip_id), None)
    if target_pos is None:
        return False, f"未找到切片: {clip_id}"

    clip = manifest.clips[target_pos]
    clip.enabled = True
    src_video = Path(manifest.source.render_video)
    clean_video = Path(manifest.source.analysis_video)
    frame_src = clean_video if clean_video.exists() else src_video
    output_dir = Path(manifest.source.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fade_ms = int(cfg.get("boundary", {}).get("audio_fade_ms", 25))
    use_vt = _has_videotoolbox()
    published_date = _ensure_published_date(manifest, store)
    prefer_two_shot = bool(cfg.get("cover", {}).get("prefer_two_shot", True))
    hero_pool = _build_hero_pool_for_manifest(frame_src, manifest, store, cfg, prefer_two_shot)

    clip.render_hash = compute_clip_render_hash(clip, manifest.source.render_video)
    fname = build_clip_filename(clip, manifest, published_date)
    out_mp4 = output_dir / fname
    cover_jpg = out_mp4.with_suffix(".jpg")
    effective_cover_title = (clip.cover_title or clip.title).strip()
    exp_cover_hash = compute_cover_hash(
        key=clip.clip_id,
        title=effective_cover_title,
        categories=clip.categories,
        start_s=clip.start_s,
        end_s=clip.end_s,
        manifest=manifest,
        cfg=cfg,
        published_date=published_date,
    )

    if not force and clip.render_hash == clip.last_rendered_hash and out_mp4.exists() and out_mp4.stat().st_size > 1024:
        clip.output_file = str(out_mp4)
        if clip.cover_hash != exp_cover_hash or not cover_jpg.exists():
            _generate_clip_cover(
                clean_video=frame_src,
                fallback_video=out_mp4,
                clip=clip,
                manifest=manifest,
                store=store,
                cfg=cfg,
                hero_pool=hero_pool,
                fallback_index=target_pos,
                published_date=published_date,
                cover_path=cover_jpg,
                reuse_cached_frame=True,
            )
            clip.cover_file = str(cover_jpg)
            clip.cover_hash = exp_cover_hash
        store.save(manifest)
        return True, str(out_mp4)

    store.log(f"[render] 单片定向渲染 {clip.clip_id}: {fname} ({clip.start_s:.2f}s → {clip.end_s:.2f}s)")
    ok, encoder, err = _render_single_clip(src_video, clip, out_mp4, fade_ms, use_vt)
    if not ok:
        store.log(f"[render] 切片 {clip.clip_id} 渲染失败: {err}")
        return False, err

    clip.last_rendered_hash = clip.render_hash
    clip.output_file = str(out_mp4)
    _generate_clip_cover(
        clean_video=frame_src,
        fallback_video=out_mp4,
        clip=clip,
        manifest=manifest,
        store=store,
        cfg=cfg,
        hero_pool=hero_pool,
        fallback_index=target_pos,
        published_date=published_date,
        cover_path=cover_jpg,
    )
    clip.cover_file = str(cover_jpg)
    clip.cover_hash = exp_cover_hash
    store.save(manifest)
    return True, str(out_mp4)


def render_compilation_and_cover(
    manifest: ProjectManifest,
    store: ProjectStore,
    cfg: dict[str, Any] | None = None,
    *,
    force: bool = False,
) -> tuple[bool, str]:
    """定向渲染切片精选集及其封面，更新 manifest 并返回结果"""
    if cfg is None:
        cfg = load_editorial_config()
    enabled_clips = [c for c in manifest.clips if c.enabled]
    if not enabled_clips:
        return False, "未选择任何切片，无法合成精选集"

    manifest.compilation.enabled = True
    use_vt = _has_videotoolbox()
    published_date = _ensure_published_date(manifest, store)
    run_id = f"comp_{uuid.uuid4().hex[:6]}"
    if force:
        manifest.compilation.last_rendered_hash = ""
    res = _render_compilation(
        manifest=manifest,
        store=store,
        cfg=cfg,
        use_vt=use_vt,
        run_id=run_id,
        published_date=published_date,
    )
    if res and res.exists():
        return True, str(res)
    return False, "精选集渲染失败，请查看日志"


