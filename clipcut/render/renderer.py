from __future__ import annotations

import functools
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import tempfile
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from ..adapters.ingest import _find_best_info_json, extract_published_date
from ..assets_gen import ensure_default_assets
from ..cancel import check_cancelled, run_cancellable_subprocess
from ..config import load_editorial_config, load_local_config
from ..models import ClipItem, ProjectManifest
from ..pipeline.content_filter import (
    build_masked_ass_file,
    compute_clean_cut_render_hash,
    describe_applied_filter,
    ensure_precise_terms_if_needed,
    get_active_cut_segments,
    get_active_cut_spans,
    get_active_delete_spans,
    get_active_mask_segments,
    get_overlapping_mask_payload,
    has_clean_source_video,
    is_visual_ocr_segment,
    merge_spans,
    resolve_mask_audio_spans,
    resolve_visual_mask_regions,
    should_use_precise_copy,
    subtract_cut_intervals,
)
from ..pipeline.verify_and_plan import (
    GENERIC_COMPILATION_TITLES,
    build_default_compilation_title,
    compute_clip_render_hash,
    compute_compilation_render_hash,
    compute_remix_render_hash,
    compute_remix_source_signature,
    get_compilation_clips,
    plan_remix_from_clips,
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

COVER_TEMPLATE_VERSION = "v2.4-smart-wrap"
REMIX_CARD_TEMPLATE_VERSION = "v3-top-nav-bar"

NAV_COLOR_PRESETS: dict[str, dict[str, Any]] = {
    "tiffany": {"label": "蒂芙尼青", "hex": "#0ABAB5", "rgb": (10, 186, 181)},
    "amber": {"label": "琥珀金", "hex": "#F59E0B", "rgb": (245, 158, 11)},
    "cobalt": {"label": "钴蓝科技", "hex": "#3B82F6", "rgb": (59, 130, 246)},
    "crimson": {"label": "绯红醒目", "hex": "#EF4444", "rgb": (239, 68, 68)},
    "emerald": {"label": "翡翠绿", "hex": "#10B981", "rgb": (16, 185, 129)},
    "violet": {"label": "紫罗兰", "hex": "#8B5CF6", "rgb": (139, 92, 246)},
}


def _normalize_hex_color(raw: str | None) -> str | None:
    s = str(raw or "").strip()
    if not s:
        return None
    if not s.startswith("#"):
        s = "#" + s
    if re.fullmatch(r"#[0-9a-fA-F]{6}", s):
        return s.upper()
    return None


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    norm = _normalize_hex_color(hex_color) or "#0ABAB5"
    return (int(norm[1:3], 16), int(norm[3:5], 16), int(norm[5:7], 16))


def _blend_rgb(c1: tuple[int, int, int], c2: tuple[int, int, int], t: float) -> tuple[int, int, int]:
    t = max(0.0, min(1.0, float(t)))
    return (
        int(round(c1[0] * (1.0 - t) + c2[0] * t)),
        int(round(c1[1] * (1.0 - t) + c2[1] * t)),
        int(round(c1[2] * (1.0 - t) + c2[2] * t)),
    )


def resolve_navigation_settings(cfg: dict[str, Any] | None = None) -> dict[str, str]:
    nav = (cfg or {}).get("navigation", {}) or {}
    valid_styles = ("top_nav", "both", "title_card")
    base_style = str(nav.get("style") or "top_nav").strip().lower()
    if base_style not in valid_styles:
        base_style = "top_nav"
    remix_style = str(nav.get("remix_style") or base_style).strip().lower()
    if remix_style not in valid_styles:
        remix_style = base_style
    compilation_style = str(nav.get("compilation_style") or base_style).strip().lower()
    if compilation_style not in valid_styles:
        compilation_style = base_style
    screen_layout = str(nav.get("screen_layout") or "portrait").strip().lower()
    if screen_layout not in ("portrait", "landscape"):
        screen_layout = "portrait"
    scheme = str(nav.get("color_scheme") or "tiffany").strip().lower()
    if scheme not in NAV_COLOR_PRESETS and scheme != "custom":
        scheme = "tiffany"
    custom_accent = _normalize_hex_color(nav.get("custom_accent")) or "#0ABAB5"
    return {
        "style": base_style,
        "remix_style": remix_style,
        "compilation_style": compilation_style,
        "screen_layout": screen_layout,
        "color_scheme": scheme,
        "custom_accent": custom_accent,
    }


def resolve_nav_palette(color_scheme: str = "tiffany", custom_accent: str = "#0ABAB5") -> dict[str, Any]:
    sc = str(color_scheme or "tiffany").strip().lower()
    if sc == "custom":
        accent_rgb = _hex_to_rgb(custom_accent)
    elif sc in NAV_COLOR_PRESETS:
        accent_rgb = NAV_COLOR_PRESETS[sc]["rgb"]
    else:
        accent_rgb = NAV_COLOR_PRESETS["tiffany"]["rgb"]

    accent_light = _blend_rgb(accent_rgb, (255, 255, 255), 0.45)
    accent_bright = _blend_rgb(accent_rgb, (255, 255, 255), 0.68)
    done_bg = _blend_rgb((16, 22, 32), accent_rgb, 0.22)
    lum = (0.299 * accent_rgb[0] + 0.587 * accent_rgb[1] + 0.114 * accent_rgb[2]) / 255.0
    active_text = (15, 23, 42, 255) if lum > 0.68 else (255, 255, 255, 255)
    active_shadow = (255, 255, 255, 70) if lum > 0.68 else (0, 20, 28, 130)
    return {
        "accent_rgb": accent_rgb,
        "accent_light": accent_light,
        "accent_bright": accent_bright,
        "done_bg": done_bg,
        "active_text": active_text,
        "active_shadow": active_shadow,
    }
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
    return sum(max(0.0, it.end_s - it.start_s) for it in enabled_items)


def build_remix_filename(manifest: ProjectManifest, published_date: str = "", duration_s: float | None = None) -> str:
    ch = _clean_file_token(manifest.metadata.channel_zh or manifest.metadata.channel or "", 24)
    title = _clean_file_token(manifest.remix.title or manifest.metadata.original_theme_zh or "高光集锦", 36)
    dt = _clean_file_token(published_date or manifest.metadata.published_date or "", 12)
    dur_val = duration_s if (duration_s is not None and duration_s > 0) else _estimate_remix_duration_s(manifest)
    parts = ["00", "Remix"] + [p for p in (ch, title, dt, _fmt_ts_token(dur_val)) if p]
    return "_".join(parts) + ".mp4"


def _resolve_compilation_effective_title(
    manifest: ProjectManifest,
    *,
    persist: bool = False,
) -> tuple[str, str]:
    """Return (display_title, effective_cover_title), filling generic placeholder via build_default_compilation_title."""
    raw_t = str(manifest.compilation.title or "").strip()
    if raw_t in GENERIC_COMPILATION_TITLES:
        disp_t = build_default_compilation_title(manifest)
        if persist and disp_t not in GENERIC_COMPILATION_TITLES:
            manifest.compilation.title = disp_t
    else:
        disp_t = raw_t
    eff_t = (manifest.compilation.cover_title or disp_t or "切片精选集").strip()
    return disp_t, eff_t


def build_compilation_filename(manifest: ProjectManifest, published_date: str = "", duration_s: float | None = None) -> str:
    ch = _clean_file_token(manifest.metadata.channel_zh or manifest.metadata.channel or "", 24)
    disp_t, _ = _resolve_compilation_effective_title(manifest, persist=False)
    title = _clean_file_token(disp_t or "切片精选集", 36)
    dt = _clean_file_token(published_date or manifest.metadata.published_date or "", 12)
    enabled_clips = get_compilation_clips(manifest)
    dur_val = duration_s if (duration_s is not None and duration_s > 0) else (
        sum(max(0.0, c.end_s - c.start_s) for c in enabled_clips)
    )
    parts = ["00", "Compilation"] + [p for p in (ch, title, dt, _fmt_ts_token(dur_val)) if p]
    return "_".join(parts) + ".mp4"


def _clean_cut_keep_spans(
    manifest: ProjectManifest,
    cfg: dict[str, Any] | None = None,
) -> list[tuple[float, float]]:
    total_dur = max(0.0, float(manifest.source.duration_s or 0.0))
    if total_dur <= 0.0 and manifest.utterances:
        total_dur = max(0.0, float(manifest.utterances[-1].end_s))
    delete_spans = get_active_delete_spans(manifest, cfg)
    return subtract_cut_intervals(0.0, total_dur, delete_spans, min_keep_s=0.4)


def build_clean_cut_filename(manifest: ProjectManifest, published_date: str = "", duration_s: float | None = None) -> str:
    ch = _clean_file_token(manifest.metadata.channel_zh or manifest.metadata.channel or "", 24)
    base_t = manifest.clean_cut.title or f"{manifest.metadata.original_theme_zh or manifest.metadata.original_title or '全片'}净版"
    title = _clean_file_token(base_t, 36) or "CleanCut"
    dt = _clean_file_token(published_date or manifest.metadata.published_date or "", 12)
    if duration_s is not None and duration_s > 0:
        dur_val = duration_s
    else:
        keep_spans = _clean_cut_keep_spans(manifest)
        dur_val = sum(max(0.0, e - s) for s, e in keep_spans) if keep_spans else float(manifest.source.duration_s or 0.0)
    parts = ["00", "CleanCut"] + [p for p in (ch, title, dt, _fmt_ts_token(dur_val)) if p]
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
    theme: str = "tiffany",
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
    if theme != "tiffany":
        payload["theme"] = theme
    if theme != "crimson":
        nav_s = resolve_navigation_settings(cfg)
        if nav_s["color_scheme"] != "tiffany":
            payload["color_scheme"] = nav_s["color_scheme"]
            if nav_s["color_scheme"] == "custom":
                payload["custom_accent"] = nav_s["custom_accent"]
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


_NO_LINE_START_CHARS = set("，,。.:：;；!！?？、)）]】》」』”’·")
_BREAK_AFTER_CHARS = set("：:，,、；;！？!?—-· ")
_OPEN_BRACKETS = set("（(【[《「")
_CLOSE_BRACKETS = set("）)】]》」")


def _wrap_title_smart(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: Any,
    max_width: int,
    max_lines: int = 2,
    stroke_width: int = 0,
) -> list[str]:
    """
    Editorial CJK/Latin title line wrapper:
    - Fits on 1 line when possible.
    - Prefers at most 2 lines; avoids 1-2 char dangling orphans on either line.
    - Never splits inside a parenthetical like （中长集锦）/（速览混剪）/（完整版）;
      when line 1 would end with an opening bracket + 1-2 chars, breaks cleanly right before the bracket.
    - Prefers semantic punctuation breaks and avoids splitting ASCII words/numbers.
    """
    s = str(text or "").strip()
    if not s:
        return [""]

    def _w(piece: str) -> float:
        if not piece:
            return 0.0
        bb = draw.textbbox((0, 0), piece, font=font, stroke_width=stroke_width)
        return float(bb[2] - bb[0])

    if _w(s) <= max_width:
        return [s]

    if max_lines <= 1:
        return _wrap_text(draw, s, font, max_width)

    # Search all valid 2-line splits (left fits <= max_width AND right fits <= max_width)
    best_split: tuple[str, str] | None = None
    best_score = -1e9
    n = len(s)

    # Precompute bracket nesting depth at each split position
    bracket_depth = [0] * (n + 1)
    depth = 0
    for idx, ch in enumerate(s):
        if ch in _OPEN_BRACKETS:
            depth += 1
        elif ch in _CLOSE_BRACKETS:
            depth = max(0, depth - 1)
        bracket_depth[idx + 1] = depth

    for i in range(1, n):
        left = s[:i].rstrip()
        right = s[i:].lstrip()
        if not left or not right:
            continue
        wl = _w(left)
        wr = _w(right)
        if wl > max_width or wr > max_width:
            continue

        score = 0.0
        # 1. Strongly avoid splitting inside brackets (e.g. "（中" + "长集锦）")
        if bracket_depth[i] > 0:
            score -= 60.0
        # 2. Strongly prefer breaking right before an opening bracket (e.g. before "（中长集锦）")
        if right[0] in _OPEN_BRACKETS:
            score += 45.0
        # 3. Never start line 2 with closing/trailing punctuation
        if right[0] in _NO_LINE_START_CHARS:
            score -= 80.0
        # 4. Never split in the middle of an ASCII word or number
        if left[-1].isascii() and left[-1].isalnum() and right[0].isascii() and right[0].isalnum():
            score -= 55.0
        # 5. Strongly avoid 1-2 char dangling orphans on either line
        if len(right) <= 2:
            score -= 50.0
        elif len(right) == 3 and right[0] not in _OPEN_BRACKETS:
            score -= 14.0
        if len(left) <= 2:
            score -= 45.0
        # 6. Bonus for breaking right after natural punctuation
        if left[-1] in _BREAK_AFTER_CHARS:
            score += 24.0
        # 7. Visual balance (slightly prefer line 1 >= line 2 editorial pyramid)
        balance_diff = abs(wl - wr) / max(1.0, float(max_width))
        score -= balance_diff * 22.0
        if wr > wl * 1.18:
            score -= 8.0

        if score > best_score:
            best_score = score
            best_split = (left, right)

    if best_split is not None:
        return [best_split[0], best_split[1]]

    # Does not fit in 2 lines at this font size — return greedy wrap (>2 lines) so caller shrinks font
    return _wrap_text(draw, s, font, max_width)


def _run_ffmpeg_encode(cmd_vt: list[str], cmd_x264: list[str], use_vt: bool) -> tuple[bool, str, str]:
    check_cancelled()
    if use_vt:
        res = run_cancellable_subprocess(cmd_vt, capture_output=True, text=True)
        if res.returncode == 0:
            return True, "h264_videotoolbox", ""
        check_cancelled()
    res2 = run_cancellable_subprocess(cmd_x264, capture_output=True, text=True)
    if res2.returncode == 0:
        return True, "libx264", ""
    return False, "failed", res2.stderr[-600:]


def _probe_stream_layout_and_packets(media_path: Path) -> tuple[int, int, list[dict[str, Any]], list[dict[str, Any]] | None]:
    out = subprocess.check_output(
        ["ffprobe", "-v", "error", "-show_entries", "stream=codec_type", "-of", "csv=p=0", str(media_path)],
        stderr=subprocess.DEVNULL,
        text=True,
    ).strip()
    types = [ln.strip() for ln in out.splitlines() if ln.strip()]
    v_cnt, a_cnt = types.count("video"), types.count("audio")
    if v_cnt != 1 or a_cnt > 1:
        raise RuntimeError(f"Precise copy requires 1 video and <=1 audio stream (got v={v_cnt}, a={a_cnt})")

    def _read_pkts(kind: str) -> list[dict[str, Any]]:
        raw = subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                f"{kind}:0",
                "-show_entries",
                "packet=pts_time,dts_time,flags",
                "-of",
                "csv=p=0",
                str(media_path),
            ],
            stderr=subprocess.PIPE,
            text=True,
        )
        pkts: list[dict[str, Any]] = []
        for ln in raw.splitlines():
            flds = ln.split(",")
            if len(flds) < 3 or not flds[0]:
                continue
            try:
                pts = float(flds[0])
                dts = float(flds[1]) if flds[1] and flds[1] != "N/A" else None
            except ValueError:
                continue
            pkts.append({"pts": pts, "dts": dts, "keyframe": "K" in flds[2]})
        return pkts

    v_pkts = _read_pkts("v")
    a_pkts = _read_pkts("a") if a_cnt == 1 else None
    if not v_pkts:
        raise RuntimeError("No video packets found for precise copy")
    return v_cnt, a_cnt, v_pkts, a_pkts


def _cut_media_precise_copy(source_video: Path, keep_spans: list[tuple[float, float]], out_path: Path) -> None:
    """Split A/V packet-accurate stream-copy matching AutoSub's remove_ads engine."""
    _, a_cnt, v_pkts, a_pkts = _probe_stream_layout_and_packets(source_video)
    plans: list[dict[str, Any]] = []
    for start_s, end_s in keep_spans:
        kf_indices = [i for i, p in enumerate(v_pkts) if p["keyframe"] and start_s <= p["pts"] < end_s]
        if not kf_indices:
            raise RuntimeError(f"No keyframe in keep span {start_s:.2f}-{end_s:.2f}")
        k_idx = kf_indices[0]
        v_start = v_pkts[k_idx]["pts"]
        v_seek = v_pkts[k_idx]["dts"] if v_pkts[k_idx]["dts"] is not None else v_start
        last_idx = k_idx - 1
        for i in range(k_idx, len(v_pkts)):
            if v_pkts[i]["pts"] >= end_s:
                break
            last_idx = i
        v_count = last_idx - k_idx + 1
        if v_count <= 0:
            raise RuntimeError(f"No video packets in keep span {start_s:.2f}-{end_s:.2f}")
        a_start = None
        a_count = 0
        if a_pkts is not None:
            a_cands = [p for p in a_pkts if p["pts"] >= v_start]
            if not a_cands:
                raise RuntimeError(f"No audio packets after {v_start:.2f}")
            a_start = a_cands[0]["pts"]
            a_count = sum(1 for p in a_pkts if a_start <= p["pts"] < end_s)
            if a_count <= 0:
                raise RuntimeError(f"Empty audio count in {a_start:.2f}-{end_s:.2f}")
        plans.append({
            "v_seek": max(0.0, v_seek - 0.01),
            "v_count": v_count,
            "a_start": a_start,
            "a_count": a_count,
        })

    ext = source_video.suffix.lower() or ".mp4"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="clipcut_precise_") as tmp:
        tmp_dir = Path(tmp)
        v_parts: list[Path] = []
        a_parts: list[Path] = []
        for idx, pl in enumerate(plans):
            check_cancelled()
            vp = tmp_dir / f"v_{idx:03d}{ext}"
            r_v = run_cancellable_subprocess(
                [
                    "ffmpeg", "-y", "-v", "error", "-seek2any", "1", "-ss", f"{pl['v_seek']:.6f}",
                    "-i", str(source_video), "-map", "0:v:0", "-frames:v", str(pl["v_count"]),
                    "-an", "-c:v", "copy", "-avoid_negative_ts", "make_zero", str(vp),
                ],
                capture_output=True,
                text=True,
            )
            if r_v.returncode != 0:
                raise RuntimeError(r_v.stderr[-300:] or "video copy failed")
            v_parts.append(vp)
            if pl["a_start"] is not None:
                ap = tmp_dir / f"a_{idx:03d}.mka"
                r_a = run_cancellable_subprocess(
                    [
                        "ffmpeg", "-y", "-v", "error", "-i", str(source_video),
                        "-ss", f"{pl['a_start']:.6f}", "-map", "0:a:0", "-frames:a", str(pl["a_count"]),
                        "-vn", "-c:a", "copy", "-avoid_negative_ts", "make_zero", str(ap),
                    ],
                    capture_output=True,
                    text=True,
                )
                if r_a.returncode != 0:
                    raise RuntimeError(r_a.stderr[-300:] or "audio copy failed")
                a_parts.append(ap)

        v_list = tmp_dir / "v.txt"
        v_list.write_text("\n".join(_format_concat_file_entry(p, tmp_dir) for p in v_parts) + "\n", encoding="utf-8")
        if a_parts:
            a_list = tmp_dir / "a.txt"
            a_list.write_text("\n".join(_format_concat_file_entry(p, tmp_dir) for p in a_parts) + "\n", encoding="utf-8")
            mux_cmd = [
                "ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", str(v_list),
                "-f", "concat", "-safe", "0", "-i", str(a_list),
                "-map", "0:v:0", "-map", "1:a:0", "-c", "copy", "-movflags", "+faststart", str(out_path),
            ]
        else:
            mux_cmd = [
                "ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", str(v_list),
                "-map", "0:v:0", "-an", "-c", "copy", "-movflags", "+faststart", str(out_path),
            ]
        r_m = run_cancellable_subprocess(mux_cmd, capture_output=True, text=True)
        if r_m.returncode != 0 or not out_path.exists() or out_path.stat().st_size <= 1024:
            raise RuntimeError(r_m.stderr[-300:] or "mux failed")


def _escape_ass_filter_path(path: Path) -> str:
    return str(path).replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")


def _span_needs_mask(ks: float, ke: float, mask_spans: list[tuple[float, float]] | None) -> bool:
    if not mask_spans or ke <= ks:
        return False
    return any(min(float(ke), float(me)) - max(float(ks), float(ms)) > 0.05 for ms, me in mask_spans)


def _span_needs_visual_mask(
    ks: float,
    ke: float,
    visual_regions: list[dict[str, Any]] | None,
) -> bool:
    if not visual_regions or ke <= ks:
        return False
    return any(
        min(float(ke), float(r.get("end_s", 0.0))) - max(float(ks), float(r.get("start_s", 0.0))) > 0.05
        and bool(r.get("boxes"))
        for r in visual_regions
    )


@functools.lru_cache(maxsize=64)
def _probe_video_wh(video_path_str: str, fallback_w: int, fallback_h: int) -> tuple[int, int]:
    try:
        out = subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height",
                "-of",
                "csv=p=0:s=x",
                video_path_str,
            ],
            text=True,
            timeout=10,
        ).strip()
        first = out.splitlines()[0].strip() if out else ""
        if "x" in first:
            ws, hs = first.split("x", 1)
            w_val, h_val = int(ws), int(hs)
            if w_val >= 32 and h_val >= 32:
                return (w_val & ~1), (h_val & ~1)
    except Exception:
        pass
    return max(32, int(fallback_w) & ~1), max(32, int(fallback_h) & ~1)


def _build_visual_blur_filter_chain(
    in_label: str,
    out_label: str,
    regions: list[dict[str, Any]],
    video_w: int,
    video_h: int,
    *,
    time_offset: float = 0.0,
    span_start: float = 0.0,
    span_end: float = 0.0,
) -> str:
    """Build chained FFmpeg split/crop/mosaic+boxblur/overlay filters for on-screen regions."""
    vw = max(32, int(video_w) & ~1)
    vh = max(32, int(video_h) & ~1)
    grouped: dict[tuple[int, int, int, int], dict[str, Any]] = {}

    for r in regions:
        rs = max(float(span_start), float(r.get("start_s", 0.0)))
        re_t = min(float(span_end), float(r.get("end_s", 0.0)))
        if re_t - rs <= 0.05:
            continue
        ls = max(0.0, rs - float(time_offset))
        le = max(ls + 0.05, re_t - float(time_offset))
        for b in r.get("boxes") or []:
            try:
                bx = float(b["x"])
                by = float(b["y"])
                bw = float(b["w"])
                bh = float(b["h"])
            except (KeyError, TypeError, ValueError):
                continue
            pw = min(vw, max(16, (int(round(bw * vw)) + 1) & ~1))
            ph = min(vh, max(16, (int(round(bh * vh)) + 1) & ~1))
            px = max(0, int(round(bx * vw)) & ~1)
            py = max(0, int(round(by * vh)) & ~1)
            if px + pw > vw:
                px = max(0, (vw - pw) & ~1)
            if py + ph > vh:
                py = max(0, (vh - ph) & ~1)
            qkey = (px // 8, py // 8, pw // 8, ph // 8)
            entry = grouped.get(qkey)
            if entry is None:
                grouped[qkey] = {"x0": px, "y0": py, "x1": px + pw, "y1": py + ph, "spans": [(ls, le)]}
            else:
                entry["x0"] = min(entry["x0"], px)
                entry["y0"] = min(entry["y0"], py)
                entry["x1"] = max(entry["x1"], px + pw)
                entry["y1"] = max(entry["y1"], py + ph)
                entry["spans"].append((ls, le))

    if not grouped:
        return f"{in_label}null{out_label}"

    stages: list[str] = [f"{in_label}format=yuv420p[vb_in]"]
    cur = "[vb_in]"
    items = list(grouped.values())
    for idx, entry in enumerate(items):
        px = max(0, int(entry["x0"]) & ~1)
        py = max(0, int(entry["y0"]) & ~1)
        pw = min(vw - px, max(16, (int(entry["x1"]) - px + 1) & ~1))
        ph = min(vh - py, max(16, (int(entry["y1"]) - py + 1) & ~1))
        spans = merge_spans(entry["spans"], max_gap_s=0.1)
        cond = "+".join(f"between(t,{s:.3f},{e:.3f})" for s, e in spans)
        mw = max(4, (pw // 14) & ~1)
        mh = max(4, (ph // 14) & ~1)
        lr = max(1, min(10, pw // 4, ph // 4))
        cr = max(1, min(6, pw // 8, ph // 8))
        nxt = out_label if idx == len(items) - 1 else f"[vb_o{idx}]"
        stages.append(f"{cur}split=2[vb_s{idx}][vb_p{idx}]")
        stages.append(
            f"[vb_p{idx}]crop=w={pw}:h={ph}:x={px}:y={py},"
            f"scale=w={mw}:h={mh}:flags=area,"
            f"scale=w={pw}:h={ph}:flags=neighbor,"
            f"boxblur=luma_radius={lr}:luma_power=2:chroma_radius={cr}:chroma_power=1[vb_b{idx}]"
        )
        stages.append(f"[vb_s{idx}][vb_b{idx}]overlay=x={px}:y={py}:enable='{cond}'{nxt}")
        cur = nxt
    return ";".join(stages)


def _prepare_mask_render_context(
    manifest: ProjectManifest,
    work_dir: Path,
    hash_key: str = "",
    cfg: dict[str, Any] | None = None,
    store: ProjectStore | None = None,
) -> tuple[Path | None, Path | None, list[tuple[float, float]], list[tuple[float, float]], list[dict[str, Any]]]:
    mask_segs = get_active_mask_segments(manifest, cfg)
    if not mask_segs:
        return None, None, [], [], []
    if store is not None:
        ensure_precise_terms_if_needed(manifest, store, cfg)
        mask_segs = get_active_mask_segments(manifest, cfg)
    visual_mask_regions = resolve_visual_mask_regions(manifest, mask_segs)
    has_clean = has_clean_source_video(manifest)
    clean_video: Path | None = Path(manifest.source.analysis_video).expanduser() if has_clean else None
    masked_ass: Path | None = (
        build_masked_ass_file(
            manifest,
            mask_segs,
            work_dir,
            hash_key=hash_key,
            allow_visual_passthrough=bool(visual_mask_regions),
        )
        if has_clean
        else None
    )
    if masked_ass is None or not masked_ass.is_file():
        clean_video = None
        masked_ass = None
    audio_mask_spans = (
        resolve_mask_audio_spans(manifest, mask_segs)
        if (clean_video is not None and masked_ass is not None)
        else []
    )
    mask_spans = merge_spans(
        [(s.start_s, s.end_s) for s in mask_segs if not is_visual_ocr_segment(s)],
        max_gap_s=0.15,
    )
    if clean_video is None and not visual_mask_regions:
        return None, None, [], [], []
    return clean_video, masked_ass, audio_mask_spans, mask_spans, visual_mask_regions


def _render_masked_span(
    video_input: Path,
    span_start: float,
    span_end: float,
    masked_ass_path: Path | None,
    audio_mask_spans: list[tuple[float, float]] | None,
    out_path: Path,
    *,
    fade_ms: int,
    use_vt: bool,
    vf_extra: str = "",
    af_prefix: str = "",
    apply_edge_fade: bool = True,
    visual_mask_regions: list[dict[str, Any]] | None = None,
    video_size: tuple[int, int] = (1920, 1080),
    overlay_png: Path | None = None,
) -> tuple[bool, str, str]:
    """Apply local visual box blur and/or burn redacted ASS onto clean video + bleep masked words."""
    start_s = max(0.0, float(span_start))
    end_s = max(start_s + 0.2, float(span_end))
    dur = max(0.2, end_s - start_s)
    local_raw = [
        (max(0.0, float(s) - start_s), min(dur, float(e) - start_s))
        for s, e in (audio_mask_spans or [])
        if float(e) > start_s and float(s) < end_s
    ]
    local_masks = merge_spans([(round(s, 3), round(e, 3)) for s, e in local_raw if e - s > 0.02], max_gap_s=0.05)

    use_ass = masked_ass_path is not None and masked_ass_path.is_file()
    vis_overlapping = [
        r
        for r in (visual_mask_regions or [])
        if min(end_s, float(r.get("end_s", 0.0))) - max(start_s, float(r.get("start_s", 0.0))) > 0.05
        and bool(r.get("boxes"))
    ]

    v_prefix = ""
    v_in_label = "[0:v]"
    if vis_overlapping:
        vw, vh = _probe_video_wh(str(video_input), video_size[0], video_size[1])
        time_off = 0.0 if use_ass else start_s
        v_prefix = _build_visual_blur_filter_chain(
            "[0:v]",
            "[vvis]",
            vis_overlapping,
            vw,
            vh,
            time_offset=time_off,
            span_start=start_s,
            span_end=end_s,
        )
        v_in_label = "[vvis]"

    use_overlay = overlay_png is not None and overlay_png.is_file()
    v_parts: list[str] = []
    if use_ass and masked_ass_path is not None:
        escaped_ass = _escape_ass_filter_path(masked_ass_path)
        v_parts.append(f"subtitles=filename='{escaped_ass}'")
        v_parts.append("setpts=PTS-STARTPTS")
    if vf_extra:
        v_parts.append(vf_extra.strip(", "))
    if use_overlay:
        if v_parts:
            v_tail = f"{v_in_label}{','.join(v_parts)}[vbase];[vbase][1:v]overlay=0:0:eof_action=repeat,format=yuv420p[vout]"
        else:
            v_tail = f"{v_in_label}[1:v]overlay=0:0:eof_action=repeat,format=yuv420p[vout]"
    else:
        v_parts.append("format=yuv420p")
        v_tail = f"{v_in_label}{','.join(v_parts)}[vout]"
    v_chain = f"{v_prefix};{v_tail}" if v_prefix else v_tail

    a_parts = ["asetpts=PTS-STARTPTS"] if use_ass else []
    if af_prefix:
        a_parts.append(af_prefix.strip(", "))
    if apply_edge_fade:
        fade_s = min(max(0.005, fade_ms / 1000.0), dur * 0.15)
        fade_out_st = max(0.0, dur - fade_s)
        a_parts.append(f"afade=t=in:st=0:d={fade_s:.3f}")
        a_parts.append(f"afade=t=out:st={fade_out_st:.3f}:d={fade_s:.3f}")

    if local_masks:
        cond = "+".join(f"between(t,{s:.3f},{e:.3f})" for s, e in local_masks)
        a_parts.append(f"volume=0:enable='{cond}'")
        a_chain = (
            f"[0:a]{','.join(a_parts)}[amuted];"
            f"sine=frequency=880:sample_rate=44100:duration={dur:.3f},"
            f"aformat=channel_layouts=stereo,volume='if(gt({cond},0),0.12,0)':eval=frame[abeep];"
            f"[amuted][abeep]amix=inputs=2:duration=first:dropout_transition=0:normalize=0[aout]"
        )
    else:
        a_filter = ",".join(a_parts) if a_parts else "anull"
        a_chain = f"[0:a]{a_filter}[aout]"

    fc = f"{v_chain};{a_chain}"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    base = [
        "ffmpeg",
        "-y",
        "-v",
        "error",
        "-ss",
        f"{start_s:.3f}",
    ]
    if use_ass:
        base.append("-copyts")
    base.extend(["-i", str(video_input)])
    if use_overlay and overlay_png is not None:
        base.extend(["-i", str(overlay_png)])
    base.extend(
        [
            "-t",
            f"{dur:.3f}",
            "-filter_complex",
            fc,
            "-map",
            "[vout]",
            "-map",
            "[aout]",
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
    )
    cmd_vt = base + ["-c:v", "h264_videotoolbox", "-q:v", "60", str(out_path)]
    cmd_x264 = base + ["-c:v", "libx264", "-preset", "fast", "-crf", "18", str(out_path)]
    return _run_ffmpeg_encode(cmd_vt, cmd_x264, use_vt)


def _render_multi_span_video(
    source_video: Path,
    keep_spans: list[tuple[float, float]],
    out_path: Path,
    *,
    fade_ms: int,
    use_vt: bool,
    precise_copy: bool,
    tmp_dir: Path,
    log_fn: Any = None,
    clean_video: Path | None = None,
    masked_ass_path: Path | None = None,
    audio_mask_spans: list[tuple[float, float]] | None = None,
    mask_spans: list[tuple[float, float]] | None = None,
    visual_mask_regions: list[dict[str, Any]] | None = None,
    video_size: tuple[int, int] = (1920, 1080),
) -> tuple[bool, str, str]:
    """Render kept time intervals into out_path (precise stream-copy or VT re-encode + concat)."""
    if not keep_spans:
        return False, "skipped", "无有效保留视频区间"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    has_text_masked_part = bool(
        clean_video is not None
        and masked_ass_path is not None
        and any(_span_needs_mask(ks, ke, mask_spans) for ks, ke in keep_spans)
    )
    has_vis_masked_part = any(_span_needs_visual_mask(ks, ke, visual_mask_regions) for ks, ke in keep_spans)
    if precise_copy and not has_text_masked_part and not has_vis_masked_part:
        try:
            _cut_media_precise_copy(source_video, keep_spans, out_path)
            return True, "precise_copy", ""
        except Exception as exc:
            if log_fn:
                log_fn(f"[render] 极速无损剪切降级为硬件重编码: {str(exc)[:140]}")

    tmp_dir.mkdir(parents=True, exist_ok=True)
    part_files: list[Path] = []
    encoder_used = "h264_videotoolbox" if use_vt else "libx264"

    for idx, (s, e) in enumerate(keep_spans):
        start_s = max(0.0, float(s))
        end_s = max(start_s + 0.2, float(e))
        seg_mp4 = tmp_dir / f"span_{uuid.uuid4().hex[:6]}_{idx:03d}.mp4"
        need_text = bool(
            clean_video is not None and masked_ass_path is not None and _span_needs_mask(start_s, end_s, mask_spans)
        )
        need_vis = _span_needs_visual_mask(start_s, end_s, visual_mask_regions)
        if need_text or need_vis:
            use_clean = clean_video is not None and masked_ass_path is not None
            ok, enc, err = _render_masked_span(
                clean_video if use_clean and clean_video is not None else source_video,
                start_s,
                end_s,
                masked_ass_path if use_clean else None,
                audio_mask_spans if use_clean else None,
                seg_mp4,
                fade_ms=fade_ms,
                use_vt=use_vt,
                apply_edge_fade=True,
                visual_mask_regions=visual_mask_regions,
                video_size=video_size,
            )
        else:
            dur = max(0.2, end_s - start_s)
            fade_s = min(max(0.005, fade_ms / 1000.0), dur * 0.15)
            fade_out_st = max(0.0, dur - fade_s)
            af = f"afade=t=in:st=0:d={fade_s:.3f},afade=t=out:st={fade_out_st:.3f}:d={fade_s:.3f}"
            base = [
                "ffmpeg", "-y", "-v", "error", "-ss", f"{start_s:.3f}", "-i", str(source_video),
                "-t", f"{dur:.3f}", "-af", af, "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-b:a", "192k", "-ar", "44100", "-ac", "2", "-movflags", "+faststart",
            ]
            cmd_vt = base + ["-c:v", "h264_videotoolbox", "-q:v", "60", str(seg_mp4)]
            cmd_x264 = base + ["-c:v", "libx264", "-preset", "fast", "-crf", "18", str(seg_mp4)]
            ok, enc, err = _run_ffmpeg_encode(cmd_vt, cmd_x264, use_vt)
        if not ok:
            return False, "failed", err
        encoder_used = enc
        part_files.append(seg_mp4)

    if len(part_files) == 1:
        part_files[0].replace(out_path)
        return True, encoder_used, ""

    concat_txt = tmp_dir / f"concat_{uuid.uuid4().hex[:6]}.txt"
    concat_txt.write_text("\n".join(_format_concat_file_entry(p, tmp_dir) for p in part_files) + "\n", encoding="utf-8")
    r_cat = run_cancellable_subprocess(
        [
            "ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0",
            "-i", str(concat_txt), "-c", "copy", "-movflags", "+faststart", str(out_path),
        ],
        capture_output=True,
        text=True,
    )
    if r_cat.returncode != 0:
        r_cat = run_cancellable_subprocess(
            [
                "ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0",
                "-i", str(concat_txt), "-c:v", "libx264", "-preset", "fast", "-crf", "18",
                "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", str(out_path),
            ],
            capture_output=True,
            text=True,
        )
    for p in part_files:
        p.unlink(missing_ok=True)
    concat_txt.unlink(missing_ok=True)
    if r_cat.returncode != 0 or not out_path.exists():
        return False, "failed", (r_cat.stderr or "concat failed")[-500:]
    return True, encoder_used, ""


def _render_single_clip(
    source_video: Path,
    clip: ClipItem,
    out_path: Path,
    fade_ms: int,
    use_vt: bool,
    *,
    cut_spans: list[tuple[float, float]] | None = None,
    tmp_dir: Path | None = None,
    clean_video: Path | None = None,
    masked_ass_path: Path | None = None,
    audio_mask_spans: list[tuple[float, float]] | None = None,
    mask_spans: list[tuple[float, float]] | None = None,
    visual_mask_regions: list[dict[str, Any]] | None = None,
    video_size: tuple[int, int] = (1920, 1080),
) -> tuple[bool, str, str]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    start_s = max(0.0, clip.start_s)
    end_s = max(start_s + 0.4, clip.end_s)
    keep_spans = subtract_cut_intervals(start_s, end_s, cut_spans, min_keep_s=0.4)
    if not keep_spans:
        return False, "skipped", "切片已被广告/政治敏感区间完全覆盖"
    needs_mask = bool(
        (
            clean_video is not None
            and masked_ass_path is not None
            and any(_span_needs_mask(ks, ke, mask_spans) for ks, ke in keep_spans)
        )
        or any(_span_needs_visual_mask(ks, ke, visual_mask_regions) for ks, ke in keep_spans)
    )
    is_trimmed = (
        len(keep_spans) > 1
        or abs(keep_spans[0][0] - round(start_s, 3)) > 0.05
        or abs(keep_spans[0][1] - round(end_s, 3)) > 0.05
    )
    if is_trimmed or needs_mask:
        local_tmp = tmp_dir or (out_path.parent / ".clipcut_span_tmp")
        return _render_multi_span_video(
            source_video,
            keep_spans,
            out_path,
            fade_ms=fade_ms,
            use_vt=use_vt,
            precise_copy=False,
            tmp_dir=local_tmp,
            clean_video=clean_video,
            masked_ass_path=masked_ass_path,
            audio_mask_spans=audio_mask_spans,
            mask_spans=mask_spans,
            visual_mask_regions=visual_mask_regions,
            video_size=video_size,
        )

    dur = max(0.5, end_s - start_s)
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
    accent_rgb: tuple[int, int, int] | None = None,
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

    if theme in ("tiffany", "crimson"):
        base_rgb = (216, 24, 24) if theme == "crimson" else (accent_rgb or (12, 174, 168))
        pill_fill = (*base_rgb, 246)
        pill_outline = (*_blend_rgb(base_rgb, (255, 255, 255), 0.90), 250)
        pill_text_shadow = (*_blend_rgb(base_rgb, (0, 0, 0), 0.72), 185)
        draw.rounded_rectangle(
            [bx + 3, by + 4, bx + bw + pad_bx * 2 + 3, by + pill_h + 4],
            radius=radius,
            fill=(0, 0, 0, 140),
        )
        draw.rounded_rectangle(
            [bx, by, bx + bw + pad_bx * 2, by + pill_h],
            radius=radius,
            fill=pill_fill,
            outline=pill_outline,
            width=3,
        )
        draw.text((bx + pad_bx - b_box[0] + 1, by + pad_by - b_box[1] + 1), badge_text, font=badge_font, fill=pill_text_shadow)
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
    if theme in ("tiffany", "crimson"):
        base_rgb = (216, 24, 24) if theme == "crimson" else (accent_rgb or (12, 174, 168))
        dur_fill = (*_blend_rgb(base_rgb, (0, 0, 0), 0.78), 236)
        dur_outline = (*_blend_rgb(base_rgb, (255, 255, 255), 0.18), 255)
        draw.rounded_rectangle(
            [dx + 3, by + 4, dx + dw + pad_bx * 2 + 3, by + pill_h + 4],
            radius=radius,
            fill=(0, 0, 0, 140),
        )
        draw.rounded_rectangle(
            [dx, by, dx + dw + pad_bx * 2, by + pill_h],
            radius=radius,
            fill=dur_fill,
            outline=dur_outline,
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
    accent_rgb: tuple[int, int, int] | None = None,
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
    is_crimson = theme == "crimson"
    is_bright_card = is_tiffany or is_crimson
    stroke_w = max(2, int(min_dim * 0.0024)) if is_bright_card else 2
    stroke_fill = (5, 44, 48, 240) if is_tiffany else ((72, 6, 10, 240) if is_crimson else (0, 0, 0, 225))
    eff_title_color = title_color or ((255, 255, 255, 255) if is_bright_card else (254, 232, 84, 255))

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
        max_lines = 2
        max_width_ratio = max(max_width_ratio, 0.92)
        min_width_ratio = max(min_width_ratio, 0.78)
        init_ratio = 0.086 if len(title) <= 12 else 0.072
        if max_title_ratio is not None:
            init_ratio = min(init_ratio, max_title_ratio)
        title_font_sz = int(w * init_ratio)

    min_font_sz = max(22, int(min_dim * 0.030))
    title_font_sz = max(min_font_sz, title_font_sz)
    pad_x_inner = max(36, int(w * 0.038))
    max_text_w = max(120, int(w * max_width_ratio) - pad_x_inner * 2)

    title_font = _load_cjk_font(title_font_sz, weight="heavy")
    while (
        len(_wrap_title_smart(probe_draw, title, title_font, max_text_w, max_lines=max_lines, stroke_width=stroke_w))
        > max_lines
        and title_font_sz > min_font_sz
    ):
        title_font_sz = max(min_font_sz, title_font_sz - 3)
        title_font = _load_cjk_font(title_font_sz, weight="heavy")
    lines = _wrap_title_smart(probe_draw, title, title_font, max_text_w, max_lines=max_lines, stroke_width=stroke_w)

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

    patch = base_img.crop((bx0, by0, bx1, by1)).filter(ImageFilter.GaussianBlur(radius=8 if is_bright_card else 10))
    if is_bright_card:
        # Luminous vertical gradient fill ('tiffany' cyan or 'crimson' red for Clean Cut)
        yg = np.linspace(0.0, 1.0, box_h, dtype=np.float32)[:, None]
        grad_arr = np.zeros((box_h, box_w, 4), dtype=np.uint8)
        top_a = float(240 if bg_alpha is None else bg_alpha)
        bot_a = float(min(252, int(top_a + 6)))
        if is_crimson:
            r0, r1 = 224.0, 168.0
            g0, g1 = 28.0, 12.0
            b0, b1 = 32.0, 18.0
            off_fill = (26, 4, 6, 225)
        elif accent_rgb is not None:
            top_rgb = _blend_rgb(accent_rgb, (255, 255, 255), 0.10)
            bot_rgb = _blend_rgb(accent_rgb, (0, 0, 0), 0.22)
            r0, r1 = float(top_rgb[0]), float(bot_rgb[0])
            g0, g1 = float(top_rgb[1]), float(bot_rgb[1])
            b0, b1 = float(top_rgb[2]), float(bot_rgb[2])
            off_fill = (*_blend_rgb(accent_rgb, (0, 0, 0), 0.86), 225)
        else:
            r0, r1 = 24.0, 10.0
            g0, g1 = 198.0, 154.0
            b0, b1 = 190.0, 152.0
            off_fill = (3, 14, 18, 225)
        grad_arr[:, :, 0] = np.clip(r0 * (1.0 - yg) + r1 * yg, 0, 255).astype(np.uint8)
        grad_arr[:, :, 1] = np.clip(g0 * (1.0 - yg) + g1 * yg, 0, 255).astype(np.uint8)
        grad_arr[:, :, 2] = np.clip(b0 * (1.0 - yg) + b1 * yg, 0, 255).astype(np.uint8)
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
            fill=off_fill,
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
    if is_bright_card and meta_parts:
        strip_y0 = max(0, min(box_h - 1, div_y_abs - by0))
        strip_layer = Image.new("RGBA", (box_w, box_h), (0, 0, 0, 0))
        strip_fill = (84, 8, 14, 175) if is_crimson else (5, 62, 66, 165)
        ImageDraw.Draw(strip_layer).rectangle([0, strip_y0, box_w, box_h], fill=strip_fill)
        frosted = Image.alpha_composite(frosted, strip_layer)

    out.paste(frosted, (bx0, by0), mask)
    draw = ImageDraw.Draw(out)

    # Frame outline + centered top accent bar
    eff_border_w = border_width if border_width is not None else (
        max(5, int(min_dim * 0.0056)) if is_bright_card else max(3, int(min_dim * 0.0038))
    )
    eff_accent = accent_rgb or (245, 158, 11)
    eff_accent_light = _blend_rgb(eff_accent, (255, 255, 255), 0.35)
    outline_col = (
        (255, 244, 244, 255)
        if is_crimson
        else ((244, 255, 254, 255) if is_tiffany else (*eff_accent, 255))
    )
    draw.rounded_rectangle(
        [bx0, by0, bx1, by1],
        radius=radius,
        outline=outline_col,
        width=eff_border_w,
    )
    pill_w = max(56, int(box_w * 0.08))
    pill_h = max(4, int(min_dim * 0.0048))
    pill_fill = (255, 255, 255, 235) if is_bright_card else (*eff_accent_light, 255)
    draw.rounded_rectangle(
        [w // 2 - pill_w // 2, by0 + 10, w // 2 + pill_w // 2, by0 + 10 + pill_h],
        radius=pill_h // 2,
        fill=pill_fill,
    )

    # Centered title lines
    y_cursor = by0 + pad_top
    shadow_col = (42, 4, 8, 205) if is_crimson else ((3, 28, 32, 200) if is_tiffany else (0, 0, 0, 210))
    sh_dx, sh_dy = (4, 4) if is_bright_card else (3, 3)
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
        div_col = (
            (255, 235, 235, 215)
            if is_crimson
            else ((235, 255, 253, 215) if is_tiffany else (*eff_accent, 155))
        )
        draw.line([(bx0 + div_margin, div_y), (bx1 - div_margin, div_y)], fill=div_col, width=2 if is_bright_card else max(1, eff_border_w // 2))
        meta_y = y_cursor + div_gap
        mb_full = draw.textbbox((0, 0), meta_text, font=meta_font)
        total_mw = mb_full[2] - mb_full[0]
        cur_x = (w - total_mw) // 2 - mb_full[0]
        my = meta_y - mb_full[1]
        sep = "  ·  "
        sep_w = draw.textbbox((0, 0), sep, font=meta_font)[2] - draw.textbbox((0, 0), sep, font=meta_font)[0]
        sep_col = (254, 240, 138, 255) if is_bright_card else (*eff_accent_light, 255)
        meta_text_col = (255, 255, 255, 255) if is_bright_card else (248, 250, 252, 255)
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
    theme: str = "tiffany",
    color_scheme: str = "tiffany",
    custom_accent: str = "#0ABAB5",
) -> Image.Image:
    """Render natural-portrait cover with guest-locked crop (or 4:3 cinema fit-width for two-shots) + themed card."""
    cover_accent = None if theme == "crimson" else resolve_nav_palette(color_scheme, custom_accent)["accent_rgb"]
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

    _draw_top_badges(bg, badge_text=badge_text, dur_text=dur_text, faces=safe_faces, theme=theme, accent_rgb=cover_accent)
    return _draw_frosted_glass_box(
        bg,
        title=title,
        meta_parts=meta_parts,
        faces=safe_faces,
        bottom_margin_ratio=0.032,
        min_width_ratio=0.64,
        max_width_ratio=0.92,
        theme=theme,
        accent_rgb=cover_accent,
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
    color_scheme: str = "amber",
    custom_accent: str = "#0ABAB5",
) -> Image.Image:
    """
    Compose the dark charcoal editorial card with customizable color scheme,
    embedding a ~52.6% area 16:9 portrait / two-shot Picture-in-Picture window on upper-right when available.
    """
    pal = resolve_nav_palette(color_scheme, custom_accent)
    accent_rgb = pal["accent_rgb"]
    accent_light = pal["accent_light"]
    accent_bright = pal["accent_bright"]
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
    if height > width:
        # Dedicated 9:16 portrait editorial title card layout
        draw = ImageDraw.Draw(canvas)
        pad_x = int(width * 0.065)
        bar_y = int(height * 0.13)
        draw.rectangle([pad_x, bar_y, pad_x + int(width * 0.10), bar_y + 7], fill=(*accent_rgb, 255))
        grp_font = _load_cjk_font(max(30, int(width * 0.036)), weight="bold")
        grp_text = f"{index_text} · {group_label}" if index_text else (group_label or "高光精选")
        draw.text((pad_x, bar_y + 22), grp_text, font=grp_font, fill=(*accent_bright, 255))
        if left_meta_lines:
            sub_font = _load_cjk_font(max(22, int(width * 0.025)), weight="medium")
            sub_str = "  ·  ".join(x for x in left_meta_lines if str(x or "").strip())
            if sub_str:
                draw.text((pad_x, bar_y + 76), sub_str, font=sub_font, fill=(168, 182, 204, 235))

        if has_pip and portrait_frame is not None:
            pip_x0 = pad_x
            pip_x1 = width - pad_x
            pip_w = pip_x1 - pip_x0
            pip_h = int(round(pip_w * 9 / 16))
            pip_y0 = int(height * 0.27)
            pip_y1 = pip_y0 + pip_h
            radius = max(16, int(width * 0.022))

            shadow = Image.new("RGBA", (width, height), (0, 0, 0, 0))
            ImageDraw.Draw(shadow).rounded_rectangle(
                [pip_x0 - 8, pip_y0 - 4, pip_x1 + 8, pip_y1 + 12],
                radius=radius + 4,
                fill=(0, 0, 0, 160),
            )
            shadow = shadow.filter(ImageFilter.GaussianBlur(radius=16))
            canvas = Image.alpha_composite(canvas, shadow)

            with Image.open(portrait_frame.image_path) as raw_im:
                pip_img = crop_pip_portrait(raw_im.convert("RGBA"), pip_w, pip_h, portrait_frame.faces)
            pip_mask = Image.new("L", (pip_w, pip_h), 0)
            ImageDraw.Draw(pip_mask).rounded_rectangle([0, 0, pip_w, pip_h], radius=radius, fill=255)
            canvas.paste(pip_img, (pip_x0, pip_y0), pip_mask)

            ImageDraw.Draw(canvas).rounded_rectangle(
                [pip_x0, pip_y0, pip_x1, pip_y1],
                radius=radius,
                outline=(*accent_rgb, 225),
                width=max(3, int(width * 0.0036)),
            )

        return _draw_frosted_glass_box(
            canvas,
            title=card_title or group_label,
            meta_parts=meta_parts,
            faces=None,
            bottom_margin_ratio=0.14,
            min_width_ratio=0.84,
            max_width_ratio=0.90,
            bg_alpha=236,
            border_width=max(3, int(min(width, height) * 0.004)),
            max_title_ratio=0.056,
            theme="obsidian",
            accent_rgb=accent_rgb,
        )

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
            outline=(*accent_rgb, 210),
            width=max(2, int(height * 0.0024)),
        )

        # Left editorial rail (x: 4.2% -> 20.0%)
        rail_x = int(width * 0.042)
        rail_w = int(width * 0.158)
        if index_text:
            idx_font = _load_cjk_font(max(36, int(height * 0.092)), weight="heavy")
            ib = draw.textbbox((0, 0), index_text, font=idx_font)
            draw.text((rail_x - ib[0], pip_y0 - ib[1]), index_text, font=idx_font, fill=(*accent_rgb, 255))
            rule_y = pip_y0 + (ib[3] - ib[1]) + int(height * 0.022)
        else:
            rule_y = pip_y0 + int(height * 0.015)
        draw.rectangle([rail_x, rule_y, rail_x + int(width * 0.036), rule_y + 5], fill=(*accent_light, 255))

        grp_font = _load_cjk_font(max(20, int(height * 0.030)), weight="bold")
        grp_lines = _wrap_text(draw, group_label or "高光精选", grp_font, rail_w)[:3]
        gy = rule_y + int(height * 0.024)
        for gl in grp_lines:
            draw.text((rail_x, gy), gl, font=grp_font, fill=(*accent_bright, 255))
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
            accent_rgb=accent_rgb,
        )

    # Fallback: classic dark charcoal editorial card without PiP
    draw = ImageDraw.Draw(canvas)
    margin_x = int(width * 0.12)
    draw.rectangle([margin_x, int(height * 0.30), margin_x + int(width * 0.055), int(height * 0.30) + 5], fill=accent_rgb)
    tag_font = _load_cjk_font(max(24, int(height * 0.032)), weight="bold")
    tag_str = f"{index_text}  /  {group_label or 'HIGHLIGHT'}" if index_text else (group_label or "HIGHLIGHT")
    draw.text((margin_x, int(height * 0.33)), tag_str, font=tag_font, fill=accent_light)
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
        accent_rgb=accent_rgb,
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
    theme: str = "tiffany",
    color_scheme: str = "tiffany",
    custom_accent: str = "#0ABAB5",
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
            color_scheme=color_scheme,
            custom_accent=custom_accent,
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
            theme=theme,
            color_scheme=color_scheme,
            custom_accent=custom_accent,
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
    theme: str = "tiffany",
    color_scheme: str = "tiffany",
    custom_accent: str = "#0ABAB5",
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
            theme=theme,
            color_scheme=color_scheme,
            custom_accent=custom_accent,
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

    nav_s = resolve_navigation_settings(cfg)
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
        color_scheme=nav_s["color_scheme"],
        custom_accent=nav_s["custom_accent"],
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
        color_scheme=nav_s["color_scheme"],
        custom_accent=nav_s["custom_accent"],
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

    nav_s = resolve_navigation_settings(cfg)
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
        color_scheme=nav_s["color_scheme"],
        custom_accent=nav_s["custom_accent"],
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
        color_scheme=nav_s["color_scheme"],
        custom_accent=nav_s["custom_accent"],
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

    enabled_clips = get_compilation_clips(manifest)
    dur_s = total_duration_s if (total_duration_s is not None and total_duration_s > 0) else sum(c.end_s - c.start_s for c in enabled_clips)
    dur_text = _fmt_dur(dur_s)
    meta_parts = _build_cover_meta_parts(manifest, published_date)
    w = manifest.source.width or 1920
    h = manifest.source.height or 1080
    _, comp_title = _resolve_compilation_effective_title(manifest, persist=True)
    badge_text = "COMPILATION  ·  精选合辑"
    left_meta = [f"总长 {dur_text}", f"精选 {len(enabled_clips)} 支切片"]

    nav_s = resolve_navigation_settings(cfg)
    out_img = _render_cover_image_at_size(
        selected=picked,
        width=w,
        height=h,
        badge_text=badge_text,
        dur_text=dur_text,
        title=comp_title,
        meta_parts=meta_parts,
        style=style,
        index_text="合辑",
        group_label="精选合辑",
        left_meta_lines=left_meta,
        color_scheme=nav_s["color_scheme"],
        custom_accent=nav_s["custom_accent"],
    )
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
        color_scheme=nav_s["color_scheme"],
        custom_accent=nav_s["custom_accent"],
    )
    return True


def _generate_clean_cut_cover(
    *,
    manifest: ProjectManifest,
    store: ProjectStore,
    cfg: dict[str, Any],
    hero_pool: list[SelectedFrame],
    published_date: str,
    cover_path: Path,
    total_duration_s: float | None = None,
) -> bool:
    """Render cover JPEG for Clean Cut full video."""
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

    keep_spans = _clean_cut_keep_spans(manifest)
    kept_dur = total_duration_s if (total_duration_s is not None and total_duration_s > 0) else (
        sum(e - s for s, e in keep_spans) if keep_spans else float(manifest.source.duration_s or 0.0)
    )
    saved_s = max(0.0, float(manifest.source.duration_s or kept_dur) - kept_dur)
    dur_text = _fmt_dur(kept_dur)
    meta_parts = _build_cover_meta_parts(manifest, published_date)
    w = manifest.source.width or 1920
    h = manifest.source.height or 1080
    cc_title = (manifest.clean_cut.cover_title or manifest.clean_cut.title or manifest.metadata.original_theme_zh or manifest.metadata.original_title or "全片净版").strip()
    badge_text = "CLEAN CUT  ·  净版全片"
    left_meta = [f"净版 {dur_text}", f"已剪除 {_fmt_dur(saved_s)}" if saved_s > 0.5 else "完整合规版"]

    out_img = _render_cover_image_at_size(
        selected=picked,
        width=w,
        height=h,
        badge_text=badge_text,
        dur_text=dur_text,
        title=cc_title,
        meta_parts=meta_parts,
        style=style,
        index_text="净版",
        group_label="净版全片",
        left_meta_lines=left_meta,
        theme="crimson",
    )
    _save_cover_jpeg(out_img, cover_path)
    sync_existing_platform_covers(
        selected=picked,
        cover_path=cover_path,
        badge_text=badge_text,
        dur_text=dur_text,
        title=cc_title,
        meta_parts=meta_parts,
        style=style,
        index_text="净版",
        group_label="净版全片",
        left_meta_lines=left_meta,
        published_date=published_date,
        hero_pool=hero_pool,
        theme="crimson",
    )
    return True



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
        enabled_clips = get_compilation_clips(manifest)
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
        _, stored_title = _resolve_compilation_effective_title(manifest, persist=False)
        dur_text = _fmt_dur(est_dur_s)
        badge_text = "COMPILATION  ·  精选合辑"
        index_text = "合辑"
        group_label = "精选合辑"
        left_meta_lines = [f"总长 {dur_text}", f"精选 {len(enabled_clips)} 支切片"]
    elif target == "clean_cut":
        keep_spans = _clean_cut_keep_spans(manifest)
        kept_dur = sum(e - s for s, e in keep_spans) if keep_spans else float(manifest.source.duration_s or 0.0)
        saved_s = max(0.0, float(manifest.source.duration_s or kept_dur) - kept_dur)
        target_mp4 = (
            Path(manifest.clean_cut.output_file)
            if manifest.clean_cut.output_file
            else (output_dir / build_clean_cut_filename(manifest, published_date, duration_s=kept_dur))
        )
        selected = _resolve_remix_selected_frame(
            manifest=manifest,
            store=store,
            hero_pool=hero_pool,
            prefer_two_shot=prefer_two_shot,
        )
        stored_title = (
            manifest.clean_cut.cover_title
            or manifest.clean_cut.title
            or manifest.metadata.original_theme_zh
            or manifest.metadata.original_title
            or "全片净版"
        ).strip()
        dur_text = _fmt_dur(kept_dur)
        badge_text = "CLEAN CUT  ·  净版全片"
        index_text = "净版"
        group_label = "净版全片"
        left_meta_lines = [f"净版 {dur_text}", f"已剪除 {_fmt_dur(saved_s)}" if saved_s > 0.5 else "完整合规版"]
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
    cover_theme = "crimson" if target == "clean_cut" else "tiffany"
    cover_jpg = target_mp4.with_suffix(".jpg")
    written: list[Path] = []
    seen_specs: set[str] = set()

    spec_keys: list[str] = []
    for plat in platforms:
        if plat == "bilibili":
            spec_keys.extend(["bili", "bili_4x3"])
        elif plat == "wechat":
            spec_keys.extend(["wechat", "wechat_share"])

    nav_s = resolve_navigation_settings(cfg)
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
            theme=cover_theme,
            color_scheme=nav_s["color_scheme"],
            custom_accent=nav_s["custom_accent"],
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
            theme=cover_theme,
            color_scheme=nav_s["color_scheme"],
            custom_accent=nav_s["custom_accent"],
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
        enabled_clips = get_compilation_clips(manifest)
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
        _, effective_title = _resolve_compilation_effective_title(manifest, persist=True)
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

    if target == "clean_cut":
        keep_spans = _clean_cut_keep_spans(manifest)
        kept_dur = sum(e - s for s, e in keep_spans) if keep_spans else float(manifest.source.duration_s or 0.0)
        target_mp4 = (
            Path(manifest.clean_cut.output_file)
            if manifest.clean_cut.output_file
            else (output_dir / build_clean_cut_filename(manifest, published_date, duration_s=kept_dur))
        )
        cover_jpg = target_mp4.with_suffix(".jpg")
        effective_title = (
            manifest.clean_cut.cover_title
            or manifest.clean_cut.title
            or manifest.metadata.original_theme_zh
            or manifest.metadata.original_title
            or "全片净版"
        ).strip()
        exp_hash = compute_cover_hash(
            key="clean_cut",
            title=effective_title,
            categories=["clean_cut"],
            start_s=0.0,
            end_s=float(manifest.source.duration_s or kept_dur),
            manifest=manifest,
            cfg=cfg,
            published_date=published_date,
            theme="crimson",
        )
        ok = _generate_clean_cut_cover(
            manifest=manifest,
            store=store,
            cfg=cfg,
            hero_pool=hero_pool,
            published_date=published_date,
            cover_path=cover_jpg,
            total_duration_s=kept_dur,
        )
        if ok:
            manifest.clean_cut.cover_file = str(cover_jpg)
            manifest.clean_cut.cover_hash = exp_hash
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


def _truncate_text_to_width(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    max_width: float,
) -> str:
    s = str(text or "").strip()
    if not s or max_width <= 8:
        return ""
    if draw.textlength(s, font=font) <= max_width:
        return s
    ell = "…"
    ell_w = draw.textlength(ell, font=font)
    if ell_w >= max_width:
        return ""
    lo, hi = 0, len(s)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if draw.textlength(s[:mid], font=font) + ell_w <= max_width:
            lo = mid
        else:
            hi = mid - 1
    return (s[:lo].rstrip() + ell) if lo > 0 else ell


def compose_nav_bar_overlay_image(
    *,
    width: int,
    height: int,
    labels: list[str],
    active_index: int,
    color_scheme: str = "tiffany",
    custom_accent: str = "#0ABAB5",
) -> Image.Image:
    """Render a persistent top chapter navigation bar RGBA strip highlighting the active segment."""
    clean_labels = [str(x or "").strip() or f"分段 {i + 1}" for i, x in enumerate(labels)]
    n = len(clean_labels)
    nav_h = max(68, int(round(height * 0.076)))
    img = Image.new("RGBA", (width, nav_h), (0, 0, 0, 0))
    if n == 0:
        return img

    pal = resolve_nav_palette(color_scheme, custom_accent)
    accent_rgb = pal["accent_rgb"]
    accent_bright = pal["accent_bright"]
    done_bg = pal["done_bg"]
    active_text = pal["active_text"]
    active_shadow = pal["active_shadow"]

    draw = ImageDraw.Draw(img)
    # Frosted dark bar background + subtle bottom hairline
    draw.rectangle([(0, 0), (width, nav_h)], fill=(11, 15, 24, 224))
    draw.line([(0, nav_h - 1), (width, nav_h - 1)], fill=(72, 86, 108, 225), width=1)

    pad_x = max(14, int(round(width * 0.011)))
    pad_y = max(8, int(round(nav_h * 0.12)))
    gap = max(7, int(round(width * 0.005)))
    usable_w = max(1, width - 2 * pad_x - gap * (n - 1))
    slot_w = usable_w / n
    radius = max(8, int(round((nav_h - 2 * pad_y) * 0.24)))

    if n <= 5:
        font_px = max(24, int(round(nav_h * 0.42)))
    elif n <= 8:
        font_px = max(21, int(round(nav_h * 0.36)))
    else:
        font_px = max(19, int(round(nav_h * 0.31)))
    font_active = _load_cjk_font(font_px, weight="bold")
    font_idle = _load_cjk_font(font_px, weight="medium")

    clamped_active = max(0, min(active_index, n - 1))
    inner_pad_x = max(10, int(round(slot_w * 0.045)))
    y0 = pad_y
    y1 = nav_h - pad_y
    text_layer = Image.new("RGBA", (width, nav_h), (0, 0, 0, 0))
    text_draw = ImageDraw.Draw(text_layer)

    for i, raw_label in enumerate(clean_labels):
        x0 = int(round(pad_x + i * (slot_w + gap)))
        x1 = int(round(pad_x + (i + 1) * slot_w + i * gap))
        if x1 <= x0 + 4:
            continue
        is_active = i == clamped_active
        is_done = i < clamped_active

        if is_active:
            draw.rounded_rectangle(
                [(x0, y0), (x1, y1)],
                radius=radius,
                fill=(*accent_rgb, 244),
                outline=(*accent_bright, 250),
                width=2,
            )
            # Crisp bottom accent bar under active tab
            draw.rectangle([(x0 + radius, nav_h - 4), (x1 - radius, nav_h - 1)], fill=(*accent_bright, 255))
            txt_color = active_text
            font = font_active
        elif is_done:
            draw.rounded_rectangle(
                [(x0, y0), (x1, y1)],
                radius=radius,
                fill=(*done_bg, 226),
            )
            draw.rectangle([(x0 + radius, nav_h - 3), (x1 - radius, nav_h - 1)], fill=(*accent_rgb, 205))
            txt_color = (232, 240, 250, 232)
            font = font_idle
        else:
            draw.rounded_rectangle(
                [(x0, y0), (x1, y1)],
                radius=radius,
                fill=(26, 33, 46, 222),
            )
            txt_color = (196, 208, 224, 215)
            font = font_idle

        prefix = f"{i + 1:02d} " if n <= 6 else f"{i + 1}."
        full_text = f"{prefix}{raw_label}"
        max_text_w = max(12.0, (x1 - x0) - 2 * inner_pad_x)
        fitted = _truncate_text_to_width(text_draw, full_text, font, max_text_w)
        if fitted:
            bbox = text_draw.textbbox((0, 0), fitted, font=font)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
            tx = x0 + ((x1 - x0) - tw) / 2.0 - bbox[0]
            ty = y0 + ((y1 - y0) - th) / 2.0 - bbox[1]
            if is_active:
                text_draw.text((tx, ty + 1), fitted, font=font, fill=active_shadow)
            text_draw.text((tx, ty), fitted, font=font, fill=txt_color)

    return Image.alpha_composite(img, text_layer)


def compose_portrait_frame_overlay_image(
    *,
    width: int = 1080,
    height: int = 1920,
    title: str,
    subtitle: str = "",
    badge_text: str = "",
    labels: list[str] | None = None,
    active_index: int = 0,
    show_nav: bool = True,
    active_meta_title: str = "",
    active_meta_badge: str = "",
    active_meta_summary: str = "",
    meta_channel: str = "",
    meta_guest: str = "",
    meta_date: str = "",
    source_duration_text: str = "",
    total_duration_text: str = "",
    color_scheme: str = "tiffany",
    custom_accent: str = "#0ABAB5",
) -> tuple[Image.Image, int, int]:
    """
    Render a 9:16 portrait editorial overlay:
    - Top black band: main title + horizontal left-to-right navigation bar flush above the video window.
    - Middle band: transparent 16:9 video window.
    - Bottom black band: structured clip/segment metadata (active segment, guest/channel/date, duration, summary).
    Returns (overlay_rgba_image, video_y, video_h).
    """
    video_h = int(round(width * 9 / 16))
    video_h -= video_h % 2
    video_y = int(round((height - video_h) * 0.49))
    video_y -= video_y % 2

    pal = resolve_nav_palette(color_scheme, custom_accent)
    accent_rgb = pal["accent_rgb"]
    accent_light = pal["accent_light"]
    accent_bright = pal["accent_bright"]
    done_bg = pal["done_bg"]
    active_text = pal["active_text"]
    active_shadow = pal["active_shadow"]

    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Top and bottom dark letterbox bands (middle [video_y, video_y + video_h) stays transparent)
    draw.rectangle([(0, 0), (width, video_y)], fill=(11, 15, 24, 255))
    draw.rectangle([(0, video_y + video_h), (width, height)], fill=(11, 15, 24, 255))
    # Subtle themed hairline borders framing the 16:9 video window
    draw.rectangle([(0, video_y - 2), (width, video_y)], fill=(*accent_rgb, 220))
    draw.rectangle([(0, video_y + video_h), (width, video_y + video_h + 2)], fill=(*accent_rgb, 220))

    text_layer = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    td = ImageDraw.Draw(text_layer)
    pad_x = int(round(width * 0.048))
    content_w = width - 2 * pad_x

    # ── Horizontal Left-to-Right Navigation Bar flush above the video window ──
    clean_labels = [str(x or "").strip() or f"分段 {i + 1}" for i, x in enumerate(labels or [])]
    nav_block_h = 0
    clamped_active = 0
    if show_nav and clean_labels:
        n = len(clean_labels)
        clamped_active = max(0, min(active_index, n - 1))
        n_rows = 1 if n <= 5 else 2
        row_h = 72 if n_rows == 1 else 56
        row_gap = 8
        nav_pad_y = 10
        nav_block_h = n_rows * row_h + (n_rows - 1) * row_gap + nav_pad_y * 2
        nav_y0 = video_y - nav_block_h - 2

        draw.rectangle([(0, nav_y0), (width, video_y - 2)], fill=(15, 20, 32, 255))
        draw.line([(0, nav_y0), (width, nav_y0)], fill=(58, 72, 96, 220), width=1)

        per_row = n if n_rows == 1 else (n + 1) // 2
        nav_pad_x = max(14, int(round(width * 0.016)))
        col_gap = 8
        radius = max(8, int(round(row_h * 0.22)))
        font_px = 26 if n_rows == 1 else 22
        if n_rows == 1 and n >= 5:
            font_px = 23
        font_active = _load_cjk_font(font_px, weight="bold")
        font_idle = _load_cjk_font(font_px, weight="medium")

        for r_idx in range(n_rows):
            row_start = r_idx * per_row
            row_end = min(n, row_start + per_row)
            row_items = clean_labels[row_start:row_end]
            if not row_items:
                continue
            m = len(row_items)
            usable_w = max(1, width - 2 * nav_pad_x - col_gap * (m - 1))
            slot_w = usable_w / m
            ry0 = nav_y0 + nav_pad_y + r_idx * (row_h + row_gap)
            ry1 = ry0 + row_h

            for c_idx, raw_label in enumerate(row_items):
                global_i = row_start + c_idx
                x0 = int(round(nav_pad_x + c_idx * (slot_w + col_gap)))
                x1 = int(round(nav_pad_x + (c_idx + 1) * slot_w + c_idx * col_gap))
                is_active = global_i == clamped_active
                is_done = global_i < clamped_active

                if is_active:
                    draw.rounded_rectangle(
                        [(x0, ry0), (x1, ry1)],
                        radius=radius,
                        fill=(*accent_rgb, 246),
                        outline=(*accent_bright, 255),
                        width=2,
                    )
                    draw.rectangle([(x0 + radius, ry1 - 4), (x1 - radius, ry1 - 1)], fill=(*accent_bright, 255))
                    txt_color = active_text
                    font = font_active
                elif is_done:
                    draw.rounded_rectangle(
                        [(x0, ry0), (x1, ry1)],
                        radius=radius,
                        fill=(*done_bg, 232),
                        outline=(*accent_rgb, 140),
                        width=1,
                    )
                    txt_color = (232, 240, 250, 235)
                    font = font_idle
                else:
                    draw.rounded_rectangle(
                        [(x0, ry0), (x1, ry1)],
                        radius=radius,
                        fill=(26, 33, 46, 232),
                        outline=(56, 68, 90, 160),
                        width=1,
                    )
                    txt_color = (196, 208, 224, 220)
                    font = font_idle

                prefix = f"{global_i + 1:02d} " if m <= 4 else f"{global_i + 1}."
                full_text = f"{prefix}{raw_label}"
                inner_pad = 10
                fitted = _truncate_text_to_width(td, full_text, font, max(16.0, (x1 - x0) - 2 * inner_pad))
                if fitted:
                    bbox = td.textbbox((0, 0), fitted, font=font)
                    tw = bbox[2] - bbox[0]
                    th = bbox[3] - bbox[1]
                    tx = x0 + ((x1 - x0) - tw) / 2.0 - bbox[0]
                    ty = ry0 + (row_h - th) / 2.0 - bbox[1]
                    if is_active:
                        td.text((tx, ty + 1), fitted, font=font, fill=active_shadow)
                    td.text((tx, ty), fitted, font=font, fill=txt_color)

    # ── Top Black Band (above nav bar): Badge + Big Bold Main Title ──
    title_area_top = int(round(height * 0.075))
    title_area_bottom = (video_y - nav_block_h - 18) if nav_block_h > 0 else (video_y - 24)
    cur_y = title_area_top
    if badge_text:
        badge_font = _load_cjk_font(max(24, int(round(width * 0.025))), weight="bold")
        bb = td.textbbox((0, 0), badge_text, font=badge_font)
        bw, bh = bb[2] - bb[0], bb[3] - bb[1]
        bpx, bpy = 18, 7
        pill_w = bw + 2 * bpx
        pill_h = bh + 2 * bpy
        bx0 = (width - pill_w) // 2
        draw.rounded_rectangle(
            [(bx0, cur_y), (bx0 + pill_w, cur_y + pill_h)],
            radius=pill_h // 2,
            fill=(*accent_rgb, 242),
            outline=(*accent_bright, 240),
            width=2,
        )
        td.text((bx0 + bpx - bb[0], cur_y + bpy - bb[1]), badge_text, font=badge_font, fill=active_text)
        cur_y += pill_h + 22

    clean_title = str(title or "").strip() or "精选合辑"
    title_avail_h = max(110, title_area_bottom - cur_y)
    title_font_px = max(48, int(round(width * 0.056)))
    title_lines: list[str] = []
    title_font = _load_cjk_font(title_font_px, weight="heavy")
    while title_font_px >= 28:
        title_font = _load_cjk_font(title_font_px, weight="heavy")
        title_lines = _wrap_title_smart(td, clean_title, title_font, content_w, max_lines=2)
        line_step = int(round(title_font_px * 1.28))
        if len(title_lines) <= 2 and len(title_lines) * line_step <= title_avail_h:
            break
        title_font_px -= 3
    title_lines = title_lines[:2]
    line_step = int(round(title_font_px * 1.28))
    total_title_h = len(title_lines) * line_step
    extra_top = max(0, (title_avail_h - total_title_h) // 2)
    ty_cursor = cur_y + extra_top
    for ln in title_lines:
        lb = td.textbbox((0, 0), ln, font=title_font)
        lw = lb[2] - lb[0]
        tx = (width - lw) // 2 - lb[0]
        ty = ty_cursor - lb[1]
        td.text((tx + 2, ty + 3), ln, font=title_font, fill=(0, 0, 0, 175))
        td.text((tx, ty), ln, font=title_font, fill=(255, 255, 255, 255))
        ty_cursor += line_step

    # ── Bottom Black Band: Horizontal Video Metadata Row + Divider + Current Segment Info ──
    meta_top = video_y + video_h + 30
    meta_bottom = height - int(round(height * 0.115))  # clear of platform bottom controls
    mx = pad_x + 8
    mw = width - 2 * mx
    my = meta_top

    # Tier 1: Horizontal metadata strip (频道 | 嘉宾 | 发布 | 原片 | 切片)
    meta_items: list[tuple[str, str, bool]] = []
    if meta_channel.strip():
        meta_items.append(("频道", meta_channel.strip(), True))
    if meta_guest.strip():
        meta_items.append(("嘉宾", meta_guest.strip(), False))
    if meta_date.strip():
        meta_items.append(("发布", meta_date.strip(), False))
    if not meta_items and subtitle.strip():
        meta_items.append(("来源", subtitle.strip(), True))
    if source_duration_text.strip():
        meta_items.append(("原片", source_duration_text.strip(), False))
    if total_duration_text.strip():
        meta_items.append(("切片", total_duration_text.strip(), False))

    if meta_items:
        lbl_font = _load_cjk_font(18, weight="regular")
        val_font = _load_cjk_font(23, weight="medium")
        col_gap = 20
        n_cols = len(meta_items)
        # Measure natural widths for proportional horizontal layout
        nat_widths: list[float] = []
        for lbl, val, _ in meta_items:
            lb = td.textbbox((0, 0), lbl, font=lbl_font)
            vb = td.textbbox((0, 0), val, font=val_font)
            nat_widths.append(max(float(lb[2] - lb[0]), float(vb[2] - vb[0]), 64.0))
        usable_w = max(120.0, float(mw - col_gap * (n_cols - 1)))
        total_nat = sum(nat_widths)
        if total_nat > usable_w:
            col_widths = [max(56.0, w_i * usable_w / total_nat) for w_i in nat_widths]
        else:
            # Spread proportionally with slight breathing room while keeping compact left-to-right flow
            extra = (usable_w - total_nat) / n_cols
            col_widths = [w_i + min(extra, 44.0) for w_i in nat_widths]

        cx = float(mx)
        for idx_m, ((lbl, val, is_accent), cw) in enumerate(zip(meta_items, col_widths)):
            if idx_m > 0:
                div_x = int(round(cx - col_gap / 2.0))
                draw.line([(div_x, my + 4), (div_x, my + 52)], fill=(255, 255, 255, 38), width=1)
            td.text((int(round(cx)), my), lbl, font=lbl_font, fill=(148, 163, 184, 210))
            val_fit = _truncate_text_to_width(td, val, val_font, max(44.0, cw - 4.0))
            val_color = (*accent_light, 245) if is_accent else (234, 240, 248, 240)
            td.text((int(round(cx)), my + 24), val_fit, font=val_font, fill=val_color)
            cx += cw + col_gap

        my += 64
        draw.line([(mx, my), (mx + mw, my)], fill=(255, 255, 255, 42), width=1)
        my += 22

    # Tier 2: Current segment info (progress + title + summary, no beat/category badge)
    seg_title = (
        str(active_meta_title or "").strip()
        or (clean_labels[clamped_active] if clean_labels else "")
        or clean_title
    )
    if seg_title:
        prefix_idx = f"{clamped_active + 1:02d}/{len(clean_labels):02d}  " if clean_labels else ""
        st_font = _load_cjk_font(32, weight="bold")
        st_lines = _wrap_text(td, f"{prefix_idx}{seg_title}", st_font, mw)[:2]
        for ln_i, ln in enumerate(st_lines):
            if ln_i == 0 and prefix_idx and ln.startswith(prefix_idx):
                p_bb = td.textbbox((0, 0), prefix_idx, font=st_font)
                p_w = p_bb[2] - p_bb[0]
                td.text((mx, my), prefix_idx, font=st_font, fill=(*accent_bright, 255))
                td.text((mx + p_w, my), ln[len(prefix_idx) :], font=st_font, fill=(255, 255, 255, 255))
            else:
                td.text((mx, my), ln, font=st_font, fill=(255, 255, 255, 255))
            my += 44
        my += 6

    if active_meta_summary and my + 30 < meta_bottom:
        sum_font = _load_cjk_font(24, weight="regular")
        max_sum_lines = max(1, min(4, (meta_bottom - my) // 34))
        sum_lines = _wrap_text(td, active_meta_summary.strip(), sum_font, mw)[:max_sum_lines]
        for ln in sum_lines:
            td.text((mx, my), ln, font=sum_font, fill=(198, 210, 226, 225))
            my += 34

    return Image.alpha_composite(img, text_layer), video_y, video_h


def _build_layout_vf(
    width: int,
    height: int,
    fps: float,
    *,
    video_y: int | None = None,
    video_h: int | None = None,
) -> str:
    if video_y is not None and video_h is not None:
        return (
            f"scale={width}:{video_h}:force_original_aspect_ratio=decrease,"
            f"pad={width}:{height}:(ow-iw)/2:{video_y}:color=#0B0F18,"
            f"fps={fps:.3f}"
        )
    return (
        f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
        f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,"
        f"fps={fps:.3f}"
    )


def _overlay_nav_bar_on_video(
    src_mp4: Path,
    nav_png: Path,
    out_mp4: Path,
    *,
    width: int,
    height: int,
    fps: float,
    use_vt: bool,
    video_y: int | None = None,
    video_h: int | None = None,
) -> tuple[bool, str, str]:
    """Composite persistent navigation/layout PNG onto a video segment with uniform stream params."""
    vf_scale = _build_layout_vf(width, height, fps, video_y=video_y, video_h=video_h)
    fc = f"[0:v]{vf_scale}[vbase];[vbase][1:v]overlay=0:0:eof_action=repeat,format=yuv420p[vout]"
    base = [
        "ffmpeg",
        "-y",
        "-v",
        "error",
        "-i",
        str(src_mp4),
        "-i",
        str(nav_png),
        "-filter_complex",
        fc,
        "-map",
        "[vout]",
        "-map",
        "0:a:0?",
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
    cmd_vt = base + ["-c:v", "h264_videotoolbox", "-q:v", "60", str(out_mp4)]
    cmd_x264 = base + ["-c:v", "libx264", "-preset", "fast", "-crf", "18", str(out_mp4)]
    return _run_ffmpeg_encode(cmd_vt, cmd_x264, use_vt)


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
    color_scheme: str = "amber",
    custom_accent: str = "#0ABAB5",
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
        color_scheme=color_scheme,
        custom_accent=custom_accent,
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
    nav_settings = resolve_navigation_settings(cfg)
    nav_style = nav_settings["remix_style"]
    screen_layout = nav_settings["screen_layout"]
    nav_color_scheme = nav_settings["color_scheme"]
    nav_custom_accent = nav_settings["custom_accent"]
    is_portrait = screen_layout == "portrait"
    has_top_nav = nav_style in ("top_nav", "both") and remix.card_mode != "none"
    has_title_card = nav_style in ("title_card", "both") and remix.card_mode != "none"
    store.log(
        f"[render] 开始精装合成 Remix（{len(enabled_items)} 小段，布局: {screen_layout} / 导航: {nav_style} / 色系: {nav_color_scheme} / 分组: {remix.card_mode}）"
    )
    tmp_dir = store.work_dir / "remix_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    src_video = Path(manifest.source.render_video)
    clean_video = Path(manifest.source.analysis_video)
    frame_src = clean_video if clean_video.exists() else src_video
    src_w, src_h = manifest.source.width or 1920, manifest.source.height or 1080
    w, h = (1080, 1920) if is_portrait else (src_w, src_h)
    fps = manifest.source.fps if 15.0 <= manifest.source.fps <= 60.0 else 30.0
    defaults = ensure_default_assets()
    sfx_path = Path(remix.sfx_path) if (remix.sfx_path and Path(remix.sfx_path).exists()) else defaults["sfx"]
    bgm_path = Path(remix.bgm_path) if (remix.bgm_path and Path(remix.bgm_path).exists()) else defaults["bgm"]
    meta_parts = _build_cover_meta_parts(manifest, published_date)

    concat_files: list[Path] = []
    card_entry_times: list[float] = []
    cursor_s = 0.0
    active_cuts = get_active_delete_spans(manifest, cfg) if manifest.clean_cut.apply_to_clips else []
    clean_vid, masked_ass, audio_mask_spans, mask_spans, visual_mask_regions = (
        _prepare_mask_render_context(
            manifest,
            store.work_dir / "masked_subs",
            hash_key=remix.render_hash or run_id,
            cfg=cfg,
            store=store,
        )
        if manifest.clean_cut.apply_to_clips
        else (None, None, [], [], [])
    )

    playable_entries: list[tuple[int, Any, list[tuple[float, float]]]] = []
    for idx, it in enumerate(enabled_items):
        sub_spans = (
            subtract_cut_intervals(it.start_s, it.end_s, active_cuts, min_keep_s=1.0)
            if active_cuts
            else [(it.start_s, it.end_s)]
        )
        if not sub_spans:
            store.log(f"[render] Remix 片段 {it.remix_id} 完全位于已屏蔽广告/敏感区间，跳过")
            continue
        playable_entries.append((idx, it, sub_spans))

    if not playable_entries:
        return None

    # Build persistent navigation bar segments when has_top_nav
    nav_labels: list[str] = []
    entry_nav_indices: list[int | None] = []
    if has_top_nav:
        last_group_nav: str | None = None
        for p_idx, (_, it, _) in enumerate(playable_entries):
            group_key = remix_item_group_key(it)
            if remix.card_mode == "every":
                nav_labels.append((it.card_title or it.group_label or f"片段 {p_idx + 1}").strip())
                entry_nav_indices.append(len(nav_labels) - 1)
            else:
                if p_idx == 0 or group_key != last_group_nav:
                    nav_labels.append((it.group_label or it.card_title or f"板块 {len(nav_labels) + 1}").strip())
                entry_nav_indices.append(len(nav_labels) - 1)
            last_group_nav = group_key
    else:
        entry_nav_indices = [None] * len(playable_entries)

    remix_disp_title = (remix.cover_title or remix.title or manifest.metadata.original_theme_zh or "高光混剪").strip()
    src_ch = (manifest.metadata.channel_zh or manifest.metadata.channel or "").strip()
    src_guest = (manifest.metadata.guest_zh or "").strip()
    remix_sub = "  ·  ".join(p for p in (src_ch, src_guest, published_date.strip()) if p)
    src_dur_txt = _fmt_dur(float(manifest.source.duration_s or 0.0)) if manifest.source.duration_s else ""
    remix_dur_txt = _fmt_dur(_estimate_remix_duration_s(manifest))
    nav_meta_by_idx: dict[int, tuple[str, str]] = {}
    for (_, it, _), n_idx in zip(playable_entries, entry_nav_indices):
        if n_idx is not None and n_idx not in nav_meta_by_idx:
            nav_meta_by_idx[n_idx] = (
                (it.card_title or it.group_label or "").strip(),
                (it.reason or it.quote or "").strip(),
            )
    nav_pngs: list[Path] = []
    portrait_static_png: Path | None = None
    portrait_vid_y: int | None = None
    portrait_vid_h: int | None = None
    if is_portrait:
        if has_top_nav and nav_labels:
            for nav_i in range(len(nav_labels)):
                png_p = tmp_dir / f"nav_portrait_{nav_i + 1:02d}.png"
                m_title, m_sum = nav_meta_by_idx.get(nav_i, (nav_labels[nav_i], ""))
                ov_img, portrait_vid_y, portrait_vid_h = compose_portrait_frame_overlay_image(
                    width=w,
                    height=h,
                    title=remix_disp_title,
                    subtitle=remix_sub,
                    badge_text="速览混剪",
                    labels=nav_labels,
                    active_index=nav_i,
                    show_nav=True,
                    active_meta_title=m_title,
                    active_meta_summary=m_sum,
                    meta_channel=src_ch,
                    meta_guest=src_guest,
                    meta_date=published_date.strip(),
                    source_duration_text=src_dur_txt,
                    total_duration_text=remix_dur_txt,
                    color_scheme=nav_color_scheme,
                    custom_accent=nav_custom_accent,
                )
                ov_img.save(png_p, format="PNG")
                nav_pngs.append(png_p)
        else:
            portrait_static_png = tmp_dir / "nav_portrait_static.png"
            ov_img, portrait_vid_y, portrait_vid_h = compose_portrait_frame_overlay_image(
                width=w,
                height=h,
                title=remix_disp_title,
                subtitle=remix_sub,
                badge_text="速览混剪",
                labels=[],
                active_index=0,
                show_nav=False,
                active_meta_title=remix_disp_title,
                active_meta_summary=remix.editorial_thesis or "",
                meta_channel=src_ch,
                meta_guest=src_guest,
                meta_date=published_date.strip(),
                source_duration_text=src_dur_txt,
                total_duration_text=remix_dur_txt,
                color_scheme=nav_color_scheme,
                custom_accent=nav_custom_accent,
            )
            ov_img.save(portrait_static_png, format="PNG")
    elif has_top_nav and nav_labels:
        for nav_i in range(len(nav_labels)):
            png_p = tmp_dir / f"nav_{nav_i + 1:02d}.png"
            compose_nav_bar_overlay_image(
                width=w,
                height=h,
                labels=nav_labels,
                active_index=nav_i,
                color_scheme=nav_color_scheme,
                custom_accent=nav_custom_accent,
            ).save(png_p, format="PNG")
            nav_pngs.append(png_p)

    prev_nav_idx: int | None = None
    last_card_group: str | None = None
    card_num = 0
    for p_idx, ((idx, it, sub_spans), nav_idx) in enumerate(zip(playable_entries, entry_nav_indices)):
        if has_title_card:
            group_key = remix_item_group_key(it)
            need_card = should_insert_title_card(remix.card_mode, p_idx, group_key, last_card_group)
            if need_card:
                card_num += 1
                card_mp4 = tmp_dir / f"card_{card_num:02d}.mp4"
                grp_start = it.start_s
                grp_end = it.end_s
                for _, nxt, _ in playable_entries[p_idx + 1 :]:
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
                ok_card = _create_title_card_video(
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
                    color_scheme=nav_color_scheme,
                    custom_accent=nav_custom_accent,
                )
                if ok_card:
                    concat_files.append(card_mp4)
                    card_entry_times.append(cursor_s)
                    cursor_s += remix.card_duration_s
            last_card_group = group_key
        elif has_top_nav and nav_idx is not None and nav_idx != prev_nav_idx:
            if cursor_s > 0.0:
                card_entry_times.append(cursor_s)
            prev_nav_idx = nav_idx

        if has_top_nav and nav_idx is not None and nav_idx < len(nav_pngs):
            nav_png = nav_pngs[nav_idx]
        elif is_portrait and portrait_static_png is not None:
            nav_png = portrait_static_png
        else:
            nav_png = None

        # Extract loudness-normalized excerpt(s) with layout & persistent navigation overlay
        for part_i, (sub_s, sub_e) in enumerate(sub_spans):
            seg_mp4 = tmp_dir / f"seg_{idx + 1:02d}_{part_i + 1:02d}.mp4"
            dur = max(0.5, sub_e - sub_s)
            vf_rmx = _build_layout_vf(
                w,
                h,
                fps,
                video_y=portrait_vid_y if is_portrait else None,
                video_h=portrait_vid_h if is_portrait else None,
            )
            need_rmx_mask = bool(
                (clean_vid is not None and masked_ass is not None and _span_needs_mask(sub_s, sub_e, mask_spans))
                or _span_needs_visual_mask(sub_s, sub_e, visual_mask_regions)
            )
            if need_rmx_mask:
                use_clean = clean_vid is not None and masked_ass is not None
                ok, _, err = _render_masked_span(
                    clean_vid if use_clean and clean_vid is not None else src_video,
                    sub_s,
                    sub_e,
                    masked_ass if use_clean else None,
                    audio_mask_spans if use_clean else None,
                    seg_mp4,
                    fade_ms=30,
                    use_vt=use_vt,
                    vf_extra=vf_rmx,
                    af_prefix="loudnorm=I=-16:TP=-1.5:LRA=11",
                    apply_edge_fade=True,
                    visual_mask_regions=visual_mask_regions,
                    video_size=(w, h),
                    overlay_png=nav_png,
                )
            else:
                fade_s = min(0.03, dur * 0.15)
                fade_out_st = max(0.0, dur - fade_s)
                af = f"loudnorm=I=-16:TP=-1.5:LRA=11,afade=t=in:st=0:d={fade_s:.3f},afade=t=out:st={fade_out_st:.3f}:d={fade_s:.3f}"

                if nav_png is not None and nav_png.is_file():
                    fc_nav = f"[0:v]{vf_rmx}[vbase];[vbase][1:v]overlay=0:0:eof_action=repeat,format=yuv420p[vout]"
                    base = [
                        "ffmpeg",
                        "-y",
                        "-v",
                        "error",
                        "-ss",
                        f"{sub_s:.3f}",
                        "-i",
                        str(src_video),
                        "-i",
                        str(nav_png),
                        "-t",
                        f"{dur:.3f}",
                        "-filter_complex",
                        fc_nav,
                        "-map",
                        "[vout]",
                        "-map",
                        "0:a:0",
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
                else:
                    vf = f"{vf_rmx},format=yuv420p"
                    base = [
                        "ffmpeg",
                        "-y",
                        "-v",
                        "error",
                        "-ss",
                        f"{sub_s:.3f}",
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
    res_cat = run_cancellable_subprocess(
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
        res_fallback = run_cancellable_subprocess(
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

        res_mix = run_cancellable_subprocess(
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
    remix.rendered_filter_tag = (
        describe_applied_filter(manifest, [(it.start_s, it.end_s) for it in enabled_items], cfg)
        if manifest.clean_cut.apply_to_clips
        else "原片直出(未关联)"
    )
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
    enabled_clips = get_compilation_clips(manifest)
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
    nav_settings = resolve_navigation_settings(cfg)
    nav_style = nav_settings["compilation_style"]
    screen_layout = nav_settings["screen_layout"]
    nav_color_scheme = nav_settings["color_scheme"]
    nav_custom_accent = nav_settings["custom_accent"]
    is_portrait = screen_layout == "portrait"
    has_top_nav = nav_style in ("top_nav", "both")
    has_title_card = nav_style in ("title_card", "both")
    card_dur = float(comp.card_duration_s or 1.8)
    card_fade = float(comp.card_fade_s or 0.2)
    est_dur_s = sum(max(0.0, c.end_s - c.start_s) for c in enabled_clips) + (
        len(enabled_clips) * card_dur if has_title_card else 0.0
    )
    out_mp4 = output_dir / build_compilation_filename(manifest, published_date, duration_s=est_dur_s)
    cover_jpg = out_mp4.with_suffix(".jpg")

    comp.render_hash = compute_compilation_render_hash(manifest)
    _, effective_title = _resolve_compilation_effective_title(manifest, persist=True)
    comp.render_hash = compute_compilation_render_hash(manifest)
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
    frame_src = Path(manifest.source.analysis_video)
    if not frame_src.exists():
        frame_src = src_video
    src_w = manifest.source.width or 1920
    src_h = manifest.source.height or 1080
    w, h = (1080, 1920) if is_portrait else (src_w, src_h)
    fps = manifest.source.fps or 30.0

    concat_files: list[Path] = []
    active_cuts = get_active_delete_spans(manifest, cfg) if manifest.clean_cut.apply_to_clips else []
    clean_vid, masked_ass, audio_mask_spans, mask_spans, visual_mask_regions = (
        _prepare_mask_render_context(
            manifest,
            store.work_dir / "masked_subs",
            hash_key=comp.render_hash or run_id,
            cfg=cfg,
            store=store,
        )
        if manifest.clean_cut.apply_to_clips
        else (None, None, [], [], [])
    )
    store.log(
        f"[render] 开始合成切片精选集 ({len(enabled_clips)} 支已选切片，布局: {screen_layout} / 导航: {nav_style} / 色系: {nav_color_scheme}，预估总长 {est_dur_s:.1f}s)"
    )

    nav_labels = [(c.cover_title or c.title).strip() or f"切片 {i}" for i, c in enumerate(enabled_clips, 1)]
    src_ch = (manifest.metadata.channel_zh or manifest.metadata.channel or "").strip()
    src_guest = (manifest.metadata.guest_zh or "").strip()
    comp_sub = "  ·  ".join(p for p in (src_ch, src_guest, published_date.strip()) if p)
    src_dur_txt = _fmt_dur(float(manifest.source.duration_s or 0.0)) if manifest.source.duration_s else ""
    for idx, clip in enumerate(enabled_clips, 1):
        # 1. Optional title card before clip (when nav_style in ("title_card", "both"))
        if has_title_card:
            card_mp4 = tmp_dir / f"card_{idx:02d}.mp4"
            card_title = (clip.cover_title or clip.title).strip()
            part_tag = (
                CATEGORY_LABELS_ZH.get(clip.categories[0], clip.categories[0])
                if clip.categories
                else (clip.chapter_title or "切片精选")
            )
            portrait = select_best_frame_for_interval(
                source_video=frame_src,
                start_s=clip.start_s,
                end_s=clip.end_s,
                work_dir=store.work_dir,
                cache_key=f"comp_{comp.render_hash}_{idx:02d}",
                hero_pool=hero_pool,
                fallback_index=idx - 1,
                prefer_two_shot=prefer_two_shot,
            )
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
                color_scheme=nav_color_scheme,
                custom_accent=nav_custom_accent,
            )
            concat_files.append(card_mp4)

        nav_png: Path | None = None
        portrait_vid_y: int | None = None
        portrait_vid_h: int | None = None
        if is_portrait:
            nav_png = tmp_dir / f"nav_portrait_{idx:02d}.png"
            ov_img, portrait_vid_y, portrait_vid_h = compose_portrait_frame_overlay_image(
                width=w,
                height=h,
                title=effective_title,
                subtitle=comp_sub,
                badge_text="中长集锦",
                labels=nav_labels,
                active_index=idx - 1,
                show_nav=has_top_nav,
                active_meta_title=(clip.cover_title or clip.title).strip(),
                active_meta_summary=(clip.summary or clip.hook or clip.key_quote or "").strip(),
                meta_channel=src_ch,
                meta_guest=src_guest,
                meta_date=published_date.strip(),
                source_duration_text=src_dur_txt,
                total_duration_text=_fmt_dur(est_dur_s),
                color_scheme=nav_color_scheme,
                custom_accent=nav_custom_accent,
            )
            ov_img.save(nav_png, format="PNG")
        elif has_top_nav:
            nav_png = tmp_dir / f"nav_{idx:02d}.png"
            compose_nav_bar_overlay_image(
                width=w,
                height=h,
                labels=nav_labels,
                active_index=idx - 1,
                color_scheme=nav_color_scheme,
                custom_accent=nav_custom_accent,
            ).save(nav_png, format="PNG")

        # 2. Obtain clean base clip segment (reuse cached standalone clip if up-to-date)
        clip_masks = (
            get_overlapping_mask_payload(clip.start_s, clip.end_s, manifest, cfg)
            if manifest.clean_cut.apply_to_clips
            else []
        )
        expected_hash = compute_clip_render_hash(
            clip,
            manifest.source.render_video,
            active_cuts,
            manifest.source.subtitle_sha256,
            mask_payload=clip_masks,
        )
        base_mp4: Path | None = (
            Path(clip.output_file)
            if (
                clip.output_file
                and clip.last_rendered_hash == expected_hash
                and Path(clip.output_file).exists()
                and Path(clip.output_file).stat().st_size > 1024
            )
            else None
        )
        if base_mp4 is None:
            raw_seg_mp4 = tmp_dir / f"raw_seg_{idx:02d}.mp4"
            fade_ms = int(cfg.get("boundary", {}).get("audio_fade_ms", 25))
            ok, _, err = _render_single_clip(
                src_video,
                clip,
                raw_seg_mp4,
                fade_ms,
                use_vt,
                cut_spans=active_cuts,
                tmp_dir=tmp_dir,
                clean_video=clean_vid,
                masked_ass_path=masked_ass,
                audio_mask_spans=audio_mask_spans,
                mask_spans=mask_spans,
                visual_mask_regions=visual_mask_regions,
                video_size=(src_w, src_h),
            )
            if ok and raw_seg_mp4.exists():
                base_mp4 = raw_seg_mp4
            else:
                store.log(f"[render] 精选集分段 {clip.clip_id} 截取失败: {err}")
                continue

        # 3. Composite layout / navigation bar overlay when portrait or has_top_nav (standalone clip stays clean)
        if nav_png is not None:
            seg_mp4 = tmp_dir / f"seg_{idx:02d}.mp4"
            ok_nav, _, err_nav = _overlay_nav_bar_on_video(
                base_mp4,
                nav_png,
                seg_mp4,
                width=w,
                height=h,
                fps=fps,
                use_vt=use_vt,
                video_y=portrait_vid_y if is_portrait else None,
                video_h=portrait_vid_h if is_portrait else None,
            )
            if ok_nav and seg_mp4.exists():
                concat_files.append(seg_mp4)
            else:
                store.log(f"[render] 精选集分段 {clip.clip_id} 叠加导航/布局失败，回退原分段: {err_nav}")
                concat_files.append(base_mp4)
        else:
            concat_files.append(base_mp4)

    if not concat_files:
        store.log("[render] 精选集无有效拼接分段")
        return None

    concat_txt = tmp_dir / "concat.txt"
    concat_txt.write_text(
        "\n".join(_format_concat_file_entry(p, tmp_dir) for p in concat_files) + "\n",
        encoding="utf-8",
    )
    raw_mp4 = tmp_dir / "compilation_raw.mp4"
    res_cat = run_cancellable_subprocess(
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
        res_fb = run_cancellable_subprocess(
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
        comp.rendered_filter_tag = (
            describe_applied_filter(manifest, [(c.start_s, c.end_s) for c in enabled_clips], cfg)
            if manifest.clean_cut.apply_to_clips
            else "原片直出(未关联)"
        )
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
        elapsed_ms = int((time.monotonic() - t0) * 1000)
        store.append_receipt(
            run_id=run_id,
            stage="render",
            actor="system",
            action="render_compilation",
            elapsed_ms=elapsed_ms,
            summary=f"切片精选集完成: {out_mp4.name}（包含 {len(enabled_clips)} 支已选切片，顶部常驻导航条 {len(enabled_clips)} 分段高亮，总长 {est_dur_s:.1f}s）",
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


def _render_clean_cut(
    manifest: ProjectManifest,
    store: ProjectStore,
    cfg: dict[str, Any],
    use_vt: bool,
    run_id: str,
    hero_pool: list[SelectedFrame] | None = None,
    published_date: str = "",
) -> Path | None:
    """Render full Clean Cut video with selected ad and/or political-sensitive spans excised or masked."""
    t0 = time.monotonic()
    cc = manifest.clean_cut
    active_segs = get_active_cut_segments(manifest, cfg)
    if not active_segs:
        cc.enabled = False
        store.log("[render] 当前没有任何生效的去广或脱敏片段，自动跳过全片净版")
        store.save(manifest)
        return None
    keep_spans = _clean_cut_keep_spans(manifest, cfg)
    if not keep_spans:
        store.log("[render] 全片净版无有效保留区间，取消渲染")
        return None

    kept_dur = sum(e - s for s, e in keep_spans)
    saved_s = max(0.0, float(manifest.source.duration_s or kept_dur) - kept_dur)
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
    out_mp4 = output_dir / build_clean_cut_filename(manifest, published_date, duration_s=kept_dur)
    cover_jpg = out_mp4.with_suffix(".jpg")

    cc.render_hash = compute_clean_cut_render_hash(manifest, cfg)
    effective_title = (
        cc.cover_title
        or cc.title
        or manifest.metadata.original_theme_zh
        or manifest.metadata.original_title
        or "全片净版"
    ).strip()
    exp_cover_hash = compute_cover_hash(
        key="clean_cut",
        title=effective_title,
        categories=["clean_cut"],
        start_s=0.0,
        end_s=float(manifest.source.duration_s or kept_dur),
        manifest=manifest,
        cfg=cfg,
        published_date=published_date,
        theme="crimson",
    )

    if cc.render_hash == cc.last_rendered_hash:
        if not (out_mp4.exists() and out_mp4.stat().st_size > 1024):
            _migrate_cached_media(cc.output_file, out_mp4, cover_jpg)
        if out_mp4.exists() and out_mp4.stat().st_size > 1024:
            cc.output_file = str(out_mp4)
            if cc.cover_hash != exp_cover_hash or not cover_jpg.exists():
                if _generate_clean_cut_cover(
                    manifest=manifest,
                    store=store,
                    cfg=cfg,
                    hero_pool=hero_pool,
                    published_date=published_date,
                    cover_path=cover_jpg,
                    total_duration_s=kept_dur,
                ):
                    cc.cover_file = str(cover_jpg)
                    cc.cover_hash = exp_cover_hash
            else:
                cc.cover_file = str(cover_jpg)
            store.save(manifest)
            store.append_receipt(
                run_id=run_id,
                stage="render",
                actor="system",
                action="render_clean_cut_cache_hit",
                summary=f"全片净版命中增量缓存（hash {cc.render_hash}），跳过重渲: {out_mp4.name}",
            )
            return out_mp4

    clean_vid, masked_ass, audio_mask_spans, mask_spans, visual_mask_regions = _prepare_mask_render_context(
        manifest,
        store.work_dir / "clean_cut_masks",
        hash_key=cc.render_hash or run_id,
        cfg=cfg,
        store=store,
    )
    ad_cnt = sum(1 for x in active_segs if x.kind == "ad")
    pol_del_cnt = sum(1 for x in active_segs if x.kind == "political_sensitive" and not is_visual_ocr_segment(x)) - len(mask_spans)
    pol_mask_cnt = len(mask_spans) + len(visual_mask_regions)
    precise_copy = should_use_precise_copy(manifest, cfg) and not mask_spans and not visual_mask_regions
    store.log(
        f"[render] 开始渲染全片净版: 生效 {len(active_segs)} 段（去广 {ad_cnt} / 删敏 {max(0, pol_del_cnt)} / 遮蔽 {pol_mask_cnt} [含画面模糊 {len(visual_mask_regions)}]，时长变化 -{saved_s:.1f}s），"
        f"模式={'极速无损' if precise_copy else '帧精准硬件重编'}"
    )

    fade_ms = int(cfg.get("boundary", {}).get("audio_fade_ms", 25))
    src_video = Path(manifest.source.render_video)
    tmp_dir = store.work_dir / "clean_cut_tmp"
    ok, encoder, err = _render_multi_span_video(
        src_video,
        keep_spans,
        out_mp4,
        fade_ms=fade_ms,
        use_vt=use_vt,
        precise_copy=precise_copy,
        tmp_dir=tmp_dir,
        log_fn=store.log,
        clean_video=clean_vid,
        masked_ass_path=masked_ass,
        audio_mask_spans=audio_mask_spans,
        mask_spans=mask_spans,
        visual_mask_regions=visual_mask_regions,
        video_size=(manifest.source.width or 1920, manifest.source.height or 1080),
    )
    if not ok or not out_mp4.exists():
        store.log(f"[render] 全片净版渲染失败: {err}")
        return None

    cc.last_rendered_hash = cc.render_hash
    cc.output_file = str(out_mp4)
    cc.rendered_filter_tag = describe_applied_filter(manifest, None, cfg)
    if _generate_clean_cut_cover(
        manifest=manifest,
        store=store,
        cfg=cfg,
        hero_pool=hero_pool,
        published_date=published_date,
        cover_path=cover_jpg,
        total_duration_s=kept_dur,
    ):
        cc.cover_file = str(cover_jpg)
        cc.cover_hash = exp_cover_hash
    store.save(manifest)

    elapsed_ms = int((time.monotonic() - t0) * 1000)
    store.append_receipt(
        run_id=run_id,
        stage="render",
        actor="system",
        action="render_clean_cut",
        elapsed_ms=elapsed_ms,
        summary=f"全片净版完成: {out_mp4.name}（保留 {_fmt_dur(kept_dur)}，剪除 {len(active_segs)} 段 / -{saved_s:.1f}s，编码: {encoder}）",
        details={
            "output_file": str(out_mp4),
            "cover_file": cc.cover_file,
            "kept_seconds": round(kept_dur, 1),
            "removed_seconds": round(saved_s, 1),
            "ads_cut": ad_cnt,
            "political_cut": max(0, pol_del_cnt),
            "political_masked": pol_mask_cnt,
            "encoder": encoder,
            "render_hash": cc.render_hash,
        },
    )
    store.log(f"[render] 全片净版完成: {out_mp4.name}")
    return out_mp4


def render_clean_cut_and_cover(
    manifest: ProjectManifest,
    store: ProjectStore,
    cfg: dict[str, Any] | None = None,
    *,
    force: bool = False,
) -> tuple[bool, str]:
    """On-demand render of Clean Cut full video and cover."""
    if cfg is None:
        cfg = load_editorial_config()
    if not get_active_cut_segments(manifest, cfg):
        manifest.clean_cut.enabled = False
        store.save(manifest)
        return False, "当前没有任何需要剪切或脱敏的内容，无需输出全片净版"
    manifest.clean_cut.enabled = True
    use_vt = _has_videotoolbox()
    published_date = _ensure_published_date(manifest, store)
    run_id = f"clean_{uuid.uuid4().hex[:6]}"
    if force:
        manifest.clean_cut.last_rendered_hash = ""
    res = _render_clean_cut(
        manifest=manifest,
        store=store,
        cfg=cfg,
        use_vt=use_vt,
        run_id=run_id,
        published_date=published_date,
    )
    if res and res.exists():
        return True, str(res)
    return False, "全片净版渲染失败，请查看日志"



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
    if manifest.compilation.output_file:
        idx_lines.append(
            f"| 00 | {manifest.compilation.title or '切片精选集'} | 切片精选集 | — | {len(get_compilation_clips(manifest))} 支串联 | `{Path(manifest.compilation.output_file).name}` |"
        )
    if manifest.clean_cut.output_file:
        kept_spans = _clean_cut_keep_spans(manifest)
        kept_s = sum(e - s for s, e in kept_spans)
        idx_lines.append(
            f"| 00 | {manifest.clean_cut.title or '全片净版'} | 全片净版 | {_fmt_dur(kept_s)} | {len(get_active_cut_segments(manifest))} 段已剪 | `{Path(manifest.clean_cut.output_file).name}` |"
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
    est_total_s = speech_total_s
    src_dur = max(1.0, float(manifest.source.duration_s or 1.0))
    ratio_pct = (est_total_s / src_dur) * 100.0 if manifest.source.duration_s else 0.0

    rmx_lines = [
        f"# {manifest.remix.title}",
        "",
        f"> 导航模式：`{manifest.remix.card_mode}`（顶部常驻 {est_cards} 分段） | 板块：{len({remix_item_group_key(x) for x in enabled_remix})} 组 | 段落：{len(enabled_remix)} 段 | 预估总长：{_fmt_dur(est_total_s)}（人声 {_fmt_dur(speech_total_s)}，占原片 {ratio_pct:.1f}%） | BGM：`{'开' if manifest.remix.enable_bgm else '关'}` | 音效：`{'开' if manifest.remix.enable_sfx else '关'}`",
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
    render_targets: list[str] | None = None,
) -> ProjectManifest:
    targets = set(render_targets) if render_targets is not None else {"clips", "remix", "compilation", "clean_cut"}
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
    apply_links = bool(manifest.clean_cut.apply_to_clips)
    active_cuts = get_active_delete_spans(manifest, cfg) if apply_links else []
    clean_vid, masked_ass, audio_mask_spans, mask_spans, visual_mask_regions = (
        _prepare_mask_render_context(
            manifest,
            store.work_dir / "masked_subs",
            hash_key=manifest.clean_cut.render_hash or run_id,
            cfg=cfg,
            store=store,
        )
        if apply_links
        else (None, None, [], [], [])
    )

    if "clips" in targets:
        for clip_pos, clip in enumerate(manifest.clips):
            check_cancelled()
            if not clip.enabled:
                continue
            clip_masks = get_overlapping_mask_payload(clip.start_s, clip.end_s, manifest, cfg) if apply_links else []
            clip.render_hash = compute_clip_render_hash(
                clip,
                manifest.source.render_video,
                active_cuts,
                manifest.source.subtitle_sha256,
                mask_payload=clip_masks,
            )
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
            ok, encoder, err = _render_single_clip(
                src_video,
                clip,
                out_mp4,
                fade_ms,
                use_vt,
                cut_spans=active_cuts,
                tmp_dir=store.work_dir / "clips_tmp",
                clean_video=clean_vid,
                masked_ass_path=masked_ass,
                audio_mask_spans=audio_mask_spans,
                mask_spans=mask_spans,
                visual_mask_regions=visual_mask_regions,
                video_size=(manifest.source.width or 1920, manifest.source.height or 1080),
            )
            celapsed = int((time.monotonic() - ct0) * 1000)

            if ok:
                clip.last_rendered_hash = clip.render_hash
                clip.output_file = str(out_mp4)
                clip.rendered_filter_tag = (
                    describe_applied_filter(manifest, [(clip.start_s, clip.end_s)], cfg)
                    if apply_links
                    else "原片直出(未关联)"
                )
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

    remix_status = ""
    comp_status = ""

    if "remix" in targets and manifest.remix.enabled:
        check_cancelled()
        has_remix_sources = bool(manifest.clips or manifest.remix_only_pool)
        cur_sig = compute_remix_source_signature(manifest)
        need_replan = has_remix_sources and (
            len(manifest.remix.items) == 0
            or (
                not manifest.remix.manual_script_locked
                and bool(manifest.remix.planned_source_signature)
                and manifest.remix.planned_source_signature != cur_sig
            )
        )
        if need_replan:
            mode_desc = "已选切片" if any(c.enabled for c in manifest.clips) else "全片自主精选"
            store.log(f"[render] AI 混剪需策展规划（模式：{mode_desc}），自动调用 AI 规划脚本...")
            if "render" in manifest.stages:
                manifest.stages["render"].message = f"AI 混剪策展中（{mode_desc}）..."
            store.save(manifest)
            manifest = plan_remix_from_clips(manifest, store, cfg, run_id)
            if "render" in manifest.stages:
                manifest.stages["render"].message = "合成 AI 混剪成片中..."
            store.save(manifest)

        if any(x.enabled for x in manifest.remix.items):
            if force:
                manifest.remix.last_rendered_hash = ""
            res_rmx = _render_remix(
                manifest,
                store,
                cfg,
                use_vt,
                run_id,
                hero_pool=hero_pool,
                published_date=published_date,
            )
            remix_status = "，AI混剪已输出" if (res_rmx and res_rmx.exists()) else "，AI混剪失败"
        elif has_remix_sources and need_replan:
            raise RuntimeError("AI 混剪未能规划出有效高光片段，请查看日志或检查模型配置")

    if "compilation" in targets and manifest.compilation.enabled and get_compilation_clips(manifest):
        check_cancelled()
        if force:
            manifest.compilation.last_rendered_hash = ""
        res_comp = _render_compilation(
            manifest,
            store,
            cfg,
            use_vt,
            run_id,
            hero_pool=hero_pool,
            published_date=published_date,
        )
        comp_status = "，精选集已输出" if (res_comp and res_comp.exists()) else "，精选集失败"

    clean_status = ""
    if "clean_cut" in targets and manifest.clean_cut.enabled and get_active_cut_segments(manifest, cfg):
        check_cancelled()
        if force:
            manifest.clean_cut.last_rendered_hash = ""
        res_clean = _render_clean_cut(
            manifest,
            store,
            cfg,
            use_vt,
            run_id,
            hero_pool=hero_pool,
            published_date=published_date,
        )
        clean_status = "，全片净版已输出" if (res_clean and res_clean.exists()) else "，全片净版失败"
    elif "clean_cut" in targets and manifest.clean_cut.enabled:
        manifest.clean_cut.enabled = False
    _write_markdown_deliverables(manifest, output_dir)

    store.save(manifest)
    total_ms = int((time.monotonic() - t0) * 1000)
    store.append_receipt(
        run_id=run_id,
        stage="render",
        actor="system",
        action="render_complete",
        elapsed_ms=total_ms,
        summary=f"渲染完成：新切 {rendered_count} 支，命中缓存 {cached_count} 支，刷新封面 {cover_refreshed} 张，失败 {failed_count} 支{remix_status}{comp_status}{clean_status}；输出目录 {output_dir}",
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
    apply_links = bool(manifest.clean_cut.apply_to_clips)
    active_cuts = get_active_delete_spans(manifest, cfg) if apply_links else []
    clean_vid, masked_ass, audio_mask_spans, mask_spans, visual_mask_regions = (
        _prepare_mask_render_context(
            manifest,
            store.work_dir / "masked_subs",
            hash_key=manifest.clean_cut.render_hash or "single",
            cfg=cfg,
            store=store,
        )
        if apply_links
        else (None, None, [], [], [])
    )

    clip_masks = get_overlapping_mask_payload(clip.start_s, clip.end_s, manifest, cfg) if apply_links else []
    clip.render_hash = compute_clip_render_hash(
        clip,
        manifest.source.render_video,
        active_cuts,
        manifest.source.subtitle_sha256,
        mask_payload=clip_masks,
    )
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
    ok, encoder, err = _render_single_clip(
        src_video,
        clip,
        out_mp4,
        fade_ms,
        use_vt,
        cut_spans=active_cuts,
        tmp_dir=store.work_dir / "clips_tmp",
        clean_video=clean_vid,
        masked_ass_path=masked_ass,
        audio_mask_spans=audio_mask_spans,
        mask_spans=mask_spans,
        visual_mask_regions=visual_mask_regions,
        video_size=(manifest.source.width or 1920, manifest.source.height or 1080),
    )
    if not ok:
        store.log(f"[render] 切片 {clip.clip_id} 渲染失败: {err}")
        return False, err

    clip.last_rendered_hash = clip.render_hash
    clip.output_file = str(out_mp4)
    clip.rendered_filter_tag = (
        describe_applied_filter(manifest, [(clip.start_s, clip.end_s)], cfg)
        if apply_links
        else "原片直出(未关联)"
    )
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
    enabled_clips = get_compilation_clips(manifest)
    if not enabled_clips:
        return False, "精选集未选择任何切片，无法合成精选集"

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
        _write_markdown_deliverables(manifest, Path(manifest.source.output_dir))
        store.save(manifest)
        return True, str(res)
    return False, "精选集渲染失败，请查看日志"


def render_remix_and_cover(
    manifest: ProjectManifest,
    store: ProjectStore,
    cfg: dict[str, Any] | None = None,
    *,
    force: bool = False,
) -> tuple[bool, str]:
    """定向渲染 AI 混剪及封面：已有段落时严格按用户当前勾选段落出片，不自动改写脚本"""
    if cfg is None:
        cfg = load_editorial_config()
    run_id = f"remix_{uuid.uuid4().hex[:6]}"
    if len(manifest.remix.items) == 0:
        has_sources = bool(manifest.clips or manifest.remix_only_pool)
        if not has_sources:
            return False, "没有可用于混剪的切片素材"
        manifest = plan_remix_from_clips(manifest, store, cfg, run_id)

    if not any(it.enabled for it in manifest.remix.items):
        return False, "未勾选任何混剪段落，请至少勾选 1 段后再渲染"

    manifest.remix.enabled = True
    use_vt = _has_videotoolbox()
    published_date = _ensure_published_date(manifest, store)
    if force:
        manifest.remix.last_rendered_hash = ""
    res = _render_remix(
        manifest=manifest,
        store=store,
        cfg=cfg,
        use_vt=use_vt,
        run_id=run_id,
        published_date=published_date,
    )
    if res and res.exists():
        _write_markdown_deliverables(manifest, Path(manifest.source.output_dir))
        store.save(manifest)
        return True, str(res)
    return False, "混剪渲染失败，请查看日志"


